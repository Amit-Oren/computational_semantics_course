"""
Bridge-Question Pipeline Runner — Both-Texts Bridging for NLI
================================================================
Four-stage pipeline:
  Stage 1: Premise + Hypothesis (both texts) → 2-4 sub-questions, at least
           one bridging/comparison question                          (LLM)
  Stage 2: Locator            — shared module, imported unchanged
  Stage 3: Answer Extractor   — shared module, imported unchanged
  Stage 4: classify_evidence  — shared module, imported unchanged

Only Stage 1 is method-specific. Stages 2-4 reuse the exact same
locate_and_answer / classify_evidence functions that p_question, h_question,
and h_multihop call — provably identical downstream code, not a fork.

Bridge-generation safeguard: if Stage 1 returns no bridging/comparison
question, retry once with an explicit instruction to add one. If it still
fails, proceed with whatever questions were generated and set
no_bridge_generated=True for error analysis — never crash on this.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from config.config import (
    DEFAULT_PARAMS,
    BridgeQuestionOutput,
    get_structured_llm,
    logger,
)
from prompts.bridge_question import (
    BRIDGE_QUESTION_GEN_SYSTEM_PROMPT,
    BRIDGE_QUESTION_GEN_USER_PROMPT,
    BRIDGE_QUESTION_RETRY_USER_PROMPT,
)
from prompts.shared_classifier import classify_evidence
from utils.locator_extractor import locate_and_answer
from utils.premise_indexer import number_sentences
from utils.retry import call_with_retry

METHOD = "bridge_question"
FALLBACK_LABEL = "Neutral"


class BridgeQuestionPipeline:
    """Both-texts bridging NLI pipeline."""

    def __init__(self, model: str, params: dict):
        self.model  = model
        self.params = params

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def stage1_generate_questions(
        self, premise: str, hypothesis: str
    ) -> BridgeQuestionOutput | None:
        messages = [
            SystemMessage(content=BRIDGE_QUESTION_GEN_SYSTEM_PROMPT),
            HumanMessage(content=BRIDGE_QUESTION_GEN_USER_PROMPT.format(
                premise=premise, hypothesis=hypothesis,
            )),
        ]
        llm = get_structured_llm(self.model, BridgeQuestionOutput, self.params)
        return call_with_retry(llm.invoke, messages)

    def stage1_retry_generate_questions(
        self, premise: str, hypothesis: str
    ) -> BridgeQuestionOutput | None:
        messages = [
            SystemMessage(content=BRIDGE_QUESTION_GEN_SYSTEM_PROMPT),
            HumanMessage(content=BRIDGE_QUESTION_RETRY_USER_PROMPT.format(
                premise=premise, hypothesis=hypothesis,
            )),
        ]
        llm = get_structured_llm(self.model, BridgeQuestionOutput, self.params)
        return call_with_retry(llm.invoke, messages)

    # ── Full sample ───────────────────────────────────────────────────────────

    def run_sample(self, sample: dict) -> dict:
        premise    = sample["premise"]
        hypothesis = sample["hypothesis"]
        warnings: list[str] = []

        # Stage 1 — generate bridging questions (sees both texts)
        q_output = None
        try:
            q_output = self.stage1_generate_questions(premise, hypothesis)
        except Exception as exc:
            logger.warning(f"Stage 1 failed: {exc}")

        if q_output is None or not q_output.questions:
            warnings.append("Stage 1 returned no questions; defaulting to Neutral.")
            return self._make_result(sample, [], FALLBACK_LABEL, warnings, no_bridge_generated=True)

        no_bridge_generated = False
        if not q_output.bridge_indices:
            warnings.append("Stage 1 produced no bridging question; retrying once.")
            retry_output = None
            try:
                retry_output = self.stage1_retry_generate_questions(premise, hypothesis)
            except Exception as exc:
                logger.warning(f"Stage 1 retry failed: {exc}")

            if retry_output and retry_output.questions:
                q_output = retry_output
                if not retry_output.bridge_indices:
                    no_bridge_generated = True
                    warnings.append("Retry still produced no bridging question.")
            else:
                no_bridge_generated = True
                warnings.append("Retry failed or returned nothing; proceeding without a bridging question.")

        questions  = q_output.questions
        bridge_set = set(q_output.bridge_indices)

        # Stage 2+3 — shared Locator + Answer Extractor per question
        indexed_sentences, numbered_premise = number_sentences(premise)
        qa_pairs = []
        for i, question in enumerate(questions):
            result = locate_and_answer(
                self.model, self.params, question,
                indexed_sentences=indexed_sentences, numbered_premise=numbered_premise,
            )
            result["is_bridge"] = i in bridge_set
            if result["answerable"]:
                qa_pairs.append(result)

        # Stage 4 — shared classifier
        if not qa_pairs:
            warnings.append("All questions unanswerable; defaulting to Neutral.")
            return self._make_result(sample, [], FALLBACK_LABEL, warnings, no_bridge_generated)

        prediction = classify_evidence(self.model, self.params, qa_pairs, hypothesis)
        return self._make_result(sample, qa_pairs, prediction, warnings, no_bridge_generated)

    @staticmethod
    def _make_result(
        sample:               dict,
        qa_pairs:             list[dict],
        prediction:           str,
        warnings:             list[str],
        no_bridge_generated:  bool,
    ) -> dict:
        return {
            "id":                   sample.get("id"),
            "premise":              sample["premise"],
            "hypothesis":           sample["hypothesis"],
            "gold_label":           sample["label"],
            "prediction":           prediction,
            "method":               METHOD,
            "qa_pairs":             qa_pairs,
            "no_bridge_generated":  no_bridge_generated,
            "warnings":             warnings,
        }


# ── Public entry point (matches runner contract used by main.py) ──────────────

def run(samples: list[dict], model: str, params: dict = DEFAULT_PARAMS) -> list[dict]:
    pipeline = BridgeQuestionPipeline(model, params)

    logger.info("=" * 60)
    logger.info("Experiment : bridge_question")
    logger.info(f"Model      : {model}")
    logger.info(f"Temperature: {params.get('temperature')}")
    logger.info(f"Max tokens : {params.get('max_tokens')}")
    logger.info(f"Samples    : {len(samples)}")
    logger.info("=" * 60)

    results = []
    skipped = 0
    no_bridge_count = 0

    for i, sample in enumerate(samples):
        sample_id = sample.get("id")
        try:
            result = pipeline.run_sample(sample)
        except Exception as exc:
            logger.error(f"[{i+1}/{len(samples)}] id={sample_id} | unexpected error: {exc}")
            skipped += 1
            continue

        if result["no_bridge_generated"]:
            no_bridge_count += 1

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
        logger.info(f"Processed        : {len(results)} samples  |  Skipped: {skipped}")
        logger.info(f"Accuracy         : {correct}/{len(results)} = {accuracy:.4f}")
        logger.info(f"No bridge generated: {no_bridge_count}/{len(results)}")
        logger.info("=" * 60)

    return results
