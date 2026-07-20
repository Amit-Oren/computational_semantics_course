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
    NLIOutput,
    get_structured_llm,
    logger,
)
from prompts.zero_shot import SYSTEM_PROMPT as ZS_SYSTEM, USER_PROMPT as ZS_USER
from prompts.h_question import (
    H_QUESTION_GEN_SYSTEM_PROMPT,
    H_QUESTION_GEN_FEW_SHOT_SYSTEM_PROMPT,
    H_QUESTION_GEN_USER_PROMPT,
)
from prompts.shared_classifier import classify_evidence
from utils.aggregation import aggregate, AGGREGATION_MODES
from utils.locator_extractor import locate_and_answer
from utils.premise_indexer import number_sentences
from utils.retry import call_with_retry
from utils.seeding import get_seeder, SEEDERS

METHOD = "h_question"
FALLBACK_LABEL = "Neutral"
DEFAULT_SEEDER = "pos"
DEFAULT_AGGREGATION = "aggregated"


class HQuestionPipeline:
    """Hypothesis-interrogation NLI pipeline."""

    def __init__(
        self,
        model: str,
        params: dict,
        seeder_name: str = DEFAULT_SEEDER,
        aggregation: str = DEFAULT_AGGREGATION,
        few_shot: bool = False,
    ):
        if seeder_name not in SEEDERS:
            raise ValueError(f"Unknown seeder '{seeder_name}'; choose from {sorted(SEEDERS)}")
        if aggregation not in AGGREGATION_MODES:
            raise ValueError(f"Unknown aggregation '{aggregation}'; choose from {AGGREGATION_MODES}")
        self.model       = model
        self.params      = params
        self.seeder_name = seeder_name
        self.aggregation = aggregation
        self.few_shot    = few_shot
        self.seeder      = get_seeder(seeder_name, model=model, params=params)

    # ── Zero-shot fallback ────────────────────────────────────────────────────

    def _zero_shot_fallback(self, sample: dict, warnings: list[str]) -> str:
        """Direct zero-shot NLI on P+H when pipeline stages produce no evidence."""
        try:
            messages = [
                SystemMessage(content=ZS_SYSTEM),
                HumanMessage(content=ZS_USER.format(
                    premise=sample["premise"], hypothesis=sample["hypothesis"],
                )),
            ]
            output = call_with_retry(get_structured_llm(self.model, NLIOutput, self.params).invoke, messages)
            if output is not None:
                warnings.append(f"Pipeline produced no evidence; zero-shot fallback predicted {output.label}.")
                return output.label
        except Exception as exc:
            logger.warning(f"Zero-shot fallback failed: {exc}")
        warnings.append("Zero-shot fallback also failed; using Neutral.")
        return FALLBACK_LABEL

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def stage1_generate_questions(
        self, hypothesis: str, keyphrases: list[str]
    ) -> HQuestionsOutput | None:
        kp_str = ", ".join(keyphrases) if keyphrases else hypothesis
        system_prompt = (
            H_QUESTION_GEN_FEW_SHOT_SYSTEM_PROMPT if self.few_shot
            else H_QUESTION_GEN_SYSTEM_PROMPT
        )
        messages = [
            SystemMessage(content=system_prompt),
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

        # Stage 0 — seed extraction (seeder-specific, no LLM for pos/svo)
        seeds = self.seeder.seed(hypothesis)
        if not seeds:
            seeds = [hypothesis]  # degenerate fallback: treat full H as one seed

        # Stage 1 — generate questions from H + seeds
        q_output = None
        try:
            q_output = self.stage1_generate_questions(hypothesis, seeds)
        except Exception as exc:
            logger.warning(f"Stage 1 failed: {exc}")

        if q_output is None or not q_output.questions:
            warnings.append("Stage 1 returned no questions; falling back to zero-shot.")
            prediction = self._zero_shot_fallback(sample, warnings)
            return self._make_result(sample, seeds, [], prediction, "", warnings)

        questions = q_output.questions

        # Stage 2 — locate + answer each question, then aggregate over survivors
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
            warnings.append("All questions unanswerable; falling back to zero-shot.")
            prediction = self._zero_shot_fallback(sample, warnings)
            return self._make_result(sample, seeds, qa_pairs, prediction, "", warnings)

        prediction, pred_reasoning = aggregate(self.aggregation, self.model, self.params, qa_pairs, hypothesis)
        return self._make_result(sample, seeds, qa_pairs, prediction, pred_reasoning, warnings)

    def _make_result(
        self,
        sample:         dict,
        seeds:          list[str],
        qa_pairs:       list[dict],
        prediction:     str,
        pred_reasoning: str,
        warnings:       list[str],
    ) -> dict:
        return {
            "id":                 sample.get("id"),
            "premise":            sample["premise"],
            "hypothesis":         sample["hypothesis"],
            "gold_label":         sample["label"],
            "prediction":         prediction,
            "prediction_reasoning": pred_reasoning,
            "qa_pairs":           qa_pairs,
            "method":             METHOD,
            "seeder":             self.seeder_name,
            "aggregation":        self.aggregation,
            "few_shot":           self.few_shot,
            "seeds":              seeds,
            "warnings":           warnings,
        }


# ── Public entry point (matches runner contract used by main.py) ──────────────

def run(
    samples: list[dict],
    model: str,
    params: dict = DEFAULT_PARAMS,
    seeder_name: str = DEFAULT_SEEDER,
    aggregation: str = DEFAULT_AGGREGATION,
    few_shot: bool = False,
) -> list[dict]:
    pipeline = HQuestionPipeline(model, params, seeder_name=seeder_name, aggregation=aggregation, few_shot=few_shot)

    logger.info("=" * 60)
    logger.info("Experiment  : h_question")
    logger.info(f"Model       : {model}")
    logger.info(f"Temperature : {params.get('temperature')}")
    logger.info(f"Max tokens  : {params.get('max_tokens')}")
    logger.info(f"Seeder      : {seeder_name}")
    logger.info(f"Aggregation : {aggregation}")
    logger.info(f"Few-shot    : {few_shot}")
    logger.info(f"Samples     : {len(samples)}")
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
