"""
Retrieve-Then-Classify — Control Baseline for NLI
==================================================
Two-stage pipeline (no question generation, no Answer Extractor):
  Stage 2: Locator    — shared module, query = the HYPOTHESIS verbatim
  Stage 4: Classifier — shared module, evidence = raw located sentences

This isolates the Locator's contribution from question generation. The
question-based methods (p_question, h_question, h_multihop, bridge_question)
all beat zero_shot — but that improvement could come from the generated
QUESTIONS, or just from the Locator narrowing 500 words down to a few
relevant sentences before classification. This control removes questions
and the Answer Extractor entirely: locate sentences relevant to the
hypothesis directly, hand the classifier those raw sentences.

    zero_shot → retrieve_then_classify gap  = value of locating alone.
    retrieve_then_classify → question methods gap = value ADDED by questions.

Stage 2 (locate_only) and Stage 4 (classify_raw_evidence) are imported
unchanged from the shared modules — see utils/locator_extractor.py and
prompts/shared_classifier.py. No new prompts: reuses the existing shared
LOCATE prompts and the shared classifier prompt.
"""

from __future__ import annotations

from config.config import DEFAULT_PARAMS, logger
from prompts.shared_classifier import classify_raw_evidence
from utils.locator_extractor import locate_only
from utils.premise_indexer import number_sentences

METHOD = "retrieve_then_classify"
FALLBACK_LABEL = "Neutral"


class RetrieveThenClassifyPipeline:
    """Locator-only control: no question generation, no Answer Extractor."""

    def __init__(self, model: str, params: dict):
        self.model  = model
        self.params = params

    def run_sample(self, sample: dict) -> dict:
        premise    = sample["premise"]
        hypothesis = sample["hypothesis"]

        # Stage 2 — shared Locator, query = the hypothesis itself
        indexed_sentences, numbered_premise = number_sentences(premise)
        loc = locate_only(
            self.model, self.params, hypothesis,
            indexed_sentences=indexed_sentences, numbered_premise=numbered_premise,
        )
        located_indices   = loc["located_indices"]
        located_sentences = loc["located_sentences"]

        # Stage 4 — shared classifier over raw located sentences (no Q/A framing)
        prediction, pred_reasoning = classify_raw_evidence(self.model, self.params, located_sentences, hypothesis)

        return self._make_result(sample, located_indices, located_sentences, prediction, pred_reasoning)

    @staticmethod
    def _make_result(
        sample:             dict,
        located_indices:    list[int],
        located_sentences:  list[str],
        prediction:         str,
        pred_reasoning:     str,
    ) -> dict:
        return {
            "id":                   sample.get("id"),
            "premise":              sample["premise"],
            "hypothesis":           sample["hypothesis"],
            "gold_label":           sample["label"],
            "prediction":           prediction,
            "prediction_reasoning": pred_reasoning,
            "method":               METHOD,
            "located_indices":      located_indices,
            "located_sentences":    located_sentences,
        }


# ── Public entry point (matches runner contract used by main.py) ──────────────

def run(samples: list[dict], model: str, params: dict = DEFAULT_PARAMS) -> list[dict]:
    pipeline = RetrieveThenClassifyPipeline(model, params)

    logger.info("=" * 60)
    logger.info("Experiment : retrieve_then_classify")
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
        logger.info(
            f"[{i+1}/{len(samples)}] id={sample_id} "
            f"| gold={result['gold_label']} | pred={result['prediction']} "
            f"| located={result['located_indices']}"
        )

    if results:
        correct  = sum(r["gold_label"] == r["prediction"] for r in results)
        accuracy = correct / len(results)
        logger.info("=" * 60)
        logger.info(f"Processed : {len(results)} samples  |  Skipped: {skipped}")
        logger.info(f"Accuracy  : {correct}/{len(results)} = {accuracy:.4f}")
        logger.info("=" * 60)

    return results
