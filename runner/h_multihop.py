"""
H-Multihop Pipeline Runner — Atomic Sub-Question Chaining for NLI
=================================================================
Four-stage pipeline:
  Stage 0:  Extract keyphrases from H (POS tagging, no LLM)
  Stage 1:  H + keyphrases → ordered 2-3 atomic sub-questions    (LLM)
  Stage 2a: Numbered premise + sub-question → sentence indices    (LLM, per hop)
  Stage 2b: Extracted sentences + prior context → hop answer      (LLM, per hop)
  Stage 3:  classify_evidence over the answered hops              (shared, LLM)

Stage 1 (decomposition) and Stage 2 (sequential locate + context-chained hop
answering) are this method's distinguishing design and are kept as-is. Only
Stage 2a's locate prompt and Stage 3's classifier are shared with q2_pipeline,
p_question, and h_question — see prompts/shared_answering.py and
prompts/shared_classifier.py.

Chain halting rule:
  If a hop returns NOT_ANSWERABLE, the chain halts there, but ALL hops
  answered up to that point still go to Stage 3 — a halted chain no longer
  auto-defaults to Neutral (see prompts/h_multihop.py CHANGELOG Fix 1).

Max chain depth = 3 (mitigates error compounding).

Dataset note: ConTRoL JSONL contains uid/premise/hypothesis/label only —
gold evidence sentence indices are not available.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from config.config import (
    DEFAULT_PARAMS,
    HMDecompOutput,
    AnswerOutput,
    LocateOutput,
    get_structured_llm,
    logger,
)
from prompts.shared_answering import (
    LOCATE_SYSTEM_PROMPT,
    LOCATE_USER_PROMPT,
)
from prompts.h_multihop import (
    HM_DECOMP_SYSTEM_PROMPT,
    HM_DECOMP_USER_PROMPT,
    HM_HOP_ANSWER_SYSTEM_PROMPT,
    HM_HOP_ANSWER_USER_PROMPT,
)
from prompts.shared_classifier import classify_evidence
from utils.pos_keyphrase import extract_keyphrases
from utils.premise_indexer import number_sentences, pull_by_indices
from utils.retry import call_with_retry

METHOD = "h_multihop"
MAX_CHAIN_DEPTH = 3
FALLBACK_LABEL  = "Neutral"
NOT_ANSWERABLE  = "NOT_ANSWERABLE"


class HMultihopPipeline:
    """Atomic sub-question chaining NLI pipeline."""

    def __init__(self, model: str, params: dict):
        self.model  = model
        self.params = params

    # ── Stage 1: Decomposition Planner ───────────────────────────────────────

    def stage1_decompose(
        self, hypothesis: str, keyphrases: list[str]
    ) -> HMDecompOutput | None:
        kp_str = ", ".join(keyphrases) if keyphrases else hypothesis
        messages = [
            SystemMessage(content=HM_DECOMP_SYSTEM_PROMPT),
            HumanMessage(content=HM_DECOMP_USER_PROMPT.format(
                hypothesis=hypothesis, keyphrases=kp_str,
            )),
        ]
        llm = get_structured_llm(self.model, HMDecompOutput, self.params)
        return call_with_retry(llm.invoke, messages)

    # ── Stage 2a: Sentence Locator (shared prompt) ───────────────────────────

    def stage2a_locate(
        self, numbered_premise: str, sub_question: str
    ) -> LocateOutput | None:
        messages = [
            SystemMessage(content=LOCATE_SYSTEM_PROMPT),
            HumanMessage(content=LOCATE_USER_PROMPT.format(
                numbered_premise=numbered_premise, question=sub_question,
            )),
        ]
        llm = get_structured_llm(self.model, LocateOutput, self.params)
        return call_with_retry(llm.invoke, messages)

    # ── Stage 2b: Per-hop Answer Extractor (kept local — needs context) ──────

    def stage2b_answer(
        self, sub_question: str, sentences: list[str], context: str
    ) -> AnswerOutput | None:
        context_block  = context if context else "(none)"
        sentences_block = "\n".join(f"- {s}" for s in sentences)
        messages = [
            SystemMessage(content=HM_HOP_ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=HM_HOP_ANSWER_USER_PROMPT.format(
                context_block=context_block,
                sub_question=sub_question,
                sentences_block=sentences_block,
            )),
        ]
        llm = get_structured_llm(self.model, AnswerOutput, self.params)
        return call_with_retry(llm.invoke, messages)

    # ── Full sample ───────────────────────────────────────────────────────────

    def run_sample(self, sample: dict) -> dict:
        premise    = sample["premise"]
        hypothesis = sample["hypothesis"]
        warnings: list[str] = []

        # Stage 0 — keyphrases (no LLM)
        kp_result  = extract_keyphrases(hypothesis)
        keyphrases = kp_result["keyphrases"]

        # Index premise sentences once
        indexed_sentences, numbered_premise = number_sentences(premise)

        # Stage 1 — decompose hypothesis into sub-questions
        decomp = None
        try:
            decomp = self.stage1_decompose(hypothesis, keyphrases)
        except Exception as exc:
            logger.warning(f"Stage 1 failed: {exc}")

        if decomp is None or not decomp.sub_questions:
            warnings.append("Stage 1 returned no sub-questions; defaulting to Neutral.")
            return self._make_result(sample, keyphrases, [], FALLBACK_LABEL, warnings)

        sub_questions = decomp.sub_questions  # already capped at 3 by validator

        # Stage 2 — sequential answering loop
        hops:    list[dict] = []
        context: str        = ""   # accumulated "Q1: ...\nA1: ...\n\n..." block

        for hop_idx, sub_q in enumerate(sub_questions):
            hop: dict = {
                "hop_index":           hop_idx,
                "sub_question":        sub_q,
                "context_at_hop":      context,
                "located_indices":     [],
                "extracted_sentences": [],
                "answer_from_P":       NOT_ANSWERABLE,
                "answerable_flag":     False,
                "halted":              False,
            }

            # Stage 2a — locate
            try:
                loc = self.stage2a_locate(numbered_premise, sub_q)
                if loc and loc.indices:
                    hop["located_indices"] = loc.indices[:5]
            except Exception as exc:
                logger.warning(f"Stage 2a failed at hop {hop_idx} ('{sub_q[:60]}'): {exc}")

            # Pull sentences
            extracted = pull_by_indices(indexed_sentences, hop["located_indices"])
            hop["extracted_sentences"] = extracted

            # Stage 2b — answer (with running context)
            if extracted:
                try:
                    ans = self.stage2b_answer(sub_q, extracted, context)
                    if ans:
                        hop["answer_from_P"]  = ans.answer
                        hop["answerable_flag"] = ans.answerable
                except Exception as exc:
                    logger.warning(f"Stage 2b failed at hop {hop_idx} ('{sub_q[:60]}'): {exc}")

            hops.append(hop)

            if not hop["answerable_flag"]:
                hop["halted"] = True
                warnings.append(
                    f"Chain halted at hop {hop_idx}: sub-question not answerable."
                )
                break

            # Extend running context for the next hop
            context += f"Q{hop_idx + 1}: {sub_q}\nA{hop_idx + 1}: {hop['answer_from_P']}\n\n"

        # Stage 3 — shared classifier over the answered hops (partial chain is fine)
        qa_pairs = [
            {
                "question":        h["sub_question"],
                "answer":          h["answer_from_P"],
                "answerable":      h["answerable_flag"],
                "located_indices": h["located_indices"],
            }
            for h in hops if h["answerable_flag"]
        ]

        if not qa_pairs:
            warnings.append("No hop was answered; defaulting to Neutral.")
            prediction = FALLBACK_LABEL
        else:
            prediction = classify_evidence(self.model, self.params, qa_pairs, hypothesis)

        return self._make_result(sample, keyphrases, qa_pairs, prediction, warnings, hops)

    @staticmethod
    def _make_result(
        sample:      dict,
        keyphrases:  list[str],
        qa_pairs:    list[dict],
        prediction:  str,
        warnings:    list[str],
        hops:        list[dict] | None = None,
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
            "chain":      hops or [],
            "warnings":   warnings,
        }


# ── Public entry point ────────────────────────────────────────────────────────

def run(samples: list[dict], model: str, params: dict = DEFAULT_PARAMS) -> list[dict]:
    pipeline = HMultihopPipeline(model, params)

    logger.info("=" * 60)
    logger.info("Experiment : h_multihop")
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
