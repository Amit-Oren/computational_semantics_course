"""
Q2 Pipeline Runner — Query-Based Factual Verification for NLI
=============================================================
Two-stage pipeline:
  Stage 1 (Question Generator): H → anchors + 2-3 verification questions
  Stage 2 (Factual Auditor):    P + H + questions → tabular decomposition
                                 + calibration flags + strict NLI label
"""

from __future__ import annotations
import time
from langchain_core.messages import SystemMessage, HumanMessage
from config.config import (
    DEFAULT_PARAMS,
    Q2QuestionOutput,
    Q2AuditOutput,
    get_structured_llm,
    logger,
)
from prompts.q2_pipeline import (
    Q2_QUESTION_SYSTEM_PROMPT,
    Q2_QUESTION_USER_PROMPT,
    Q2_AUDIT_SYSTEM_PROMPT,
    Q2_AUDIT_USER_PROMPT,
)


_RETRY_WAIT   = 30   # seconds between retries on capacity errors
_MAX_RETRIES  = 5    # max attempts per LLM call


def _call_with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying on capacity/rate-limit errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            is_capacity = any(k in msg for k in ("capacity", "rate limit", "429", "503", "overloaded"))
            if is_capacity and attempt < _MAX_RETRIES:
                logger.warning(f"Provider at capacity (attempt {attempt}/{_MAX_RETRIES}) — retrying in {_RETRY_WAIT}s")
                time.sleep(_RETRY_WAIT)
            else:
                raise
    return None


class Q2Pipeline:
    """Two-stage NLI verifier: question generation → factual audit."""

    def __init__(self, model: str, params: dict):
        self.model = model
        self.params = params

    def stage1_generate_questions(self, hypothesis: str) -> Q2QuestionOutput | None:
        messages = [
            SystemMessage(content=Q2_QUESTION_SYSTEM_PROMPT),
            HumanMessage(content=Q2_QUESTION_USER_PROMPT.format(hypothesis=hypothesis)),
        ]
        llm = get_structured_llm(self.model, Q2QuestionOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    def stage2_audit(
        self,
        premise: str,
        hypothesis: str,
        stage1_output: Q2QuestionOutput,
    ) -> Q2AuditOutput | None:
        questions_block = "\n".join(
            f"{i + 1}. {q}" for i, q in enumerate(stage1_output.questions)
        )
        messages = [
            SystemMessage(content=Q2_AUDIT_SYSTEM_PROMPT),
            HumanMessage(content=Q2_AUDIT_USER_PROMPT.format(
                premise=premise,
                hypothesis=hypothesis,
                questions=questions_block,
            )),
        ]
        llm = get_structured_llm(self.model, Q2AuditOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    def run_sample(self, sample: dict) -> dict | None:
        """Run both stages for one sample. Returns None if either stage fails."""
        q_output = self.stage1_generate_questions(sample["hypothesis"])
        if q_output is None:
            return None

        a_output = self.stage2_audit(sample["premise"], sample["hypothesis"], q_output)
        if a_output is None:
            return None

        return {
            "id":         sample.get("id"),
            "premise":    sample["premise"],
            "hypothesis": sample["hypothesis"],
            "label":      sample["label"],
            "prediction": a_output.label,
            "stage1_anchors":   q_output.anchors,
            "stage1_questions": q_output.questions,
            "stage2_audit": {
                "audit_table_decomposition": [
                    {
                        "question":                       row.question,
                        "target_anchor":                  row.target_anchor,
                        "verbatim_premise_evidence_list": row.verbatim_premise_evidence_list,
                        "integrated_premise_tags":        row.integrated_premise_tags,
                        "found":                          row.found,
                    }
                    for row in a_output.audit_table_decomposition
                ],
                "matrix_cross_check_flags": a_output.matrix_cross_check_flags,
                "label":       a_output.label,
                "explanation": a_output.explanation,
            },
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

        if result is None:
            logger.warning(
                f"[{i+1}/{len(samples)}] id={sample_id} | one or both stages failed, skipping"
            )
            skipped += 1
            continue

        results.append(result)
        logger.info(
            f"[{i+1}/{len(samples)}] id={sample_id} "
            f"| gold={result['label']} | pred={result['prediction']}"
        )

    if results:
        correct = sum(r["label"] == r["prediction"] for r in results)
        accuracy = correct / len(results)
        logger.info("=" * 60)
        logger.info(f"Processed : {len(results)} samples  |  Skipped: {skipped}")
        logger.info(f"Accuracy  : {correct}/{len(results)} = {accuracy:.4f}")
        logger.info("=" * 60)

    return results
