"""
H-Multihop Pipeline Runner — Atomic Sub-Question Chaining for NLI
=================================================================
Four-stage pipeline:
  Stage 0:  Extract keyphrases from H (POS tagging, no LLM)
  Stage 1:  H + keyphrases → ordered 2-3 atomic sub-questions    (LLM)
  Stage 2a: Numbered premise + sub-question → sentence indices    (LLM, per hop)
  Stage 2b: Extracted sentences + prior context → hop answer      (LLM, per hop)
  Stage 3:  Full evidence chain → holistic NLI label              (LLM)

Chain halting rule:
  If any hop returns NOT_ANSWERABLE → halt chain → Neutral (no Stage 3 call).
  If chain completes → Stage 3 classifier decides Entailment/Contradiction/Neutral.

Max chain depth = 3 (mitigates error compounding).

Dataset note: ConTRoL JSONL contains uid/premise/hypothesis/label only —
gold evidence sentence indices are not available.
"""

from __future__ import annotations

import time
from langchain_core.messages import SystemMessage, HumanMessage

from config.config import (
    DEFAULT_PARAMS,
    HMDecompOutput,
    HMAnswerOutput,
    HLocateOutput,
    HCompareOutput,
    get_structured_llm,
    logger,
)
from prompts.h_question import (
    H_LOCATE_SYSTEM_PROMPT,
    H_LOCATE_USER_PROMPT,
)
from prompts.h_multihop import (
    HM_DECOMP_SYSTEM_PROMPT,
    HM_DECOMP_USER_PROMPT,
    HM_HOP_ANSWER_SYSTEM_PROMPT,
    HM_HOP_ANSWER_USER_PROMPT,
    HM_CLASSIFY_SYSTEM_PROMPT,
    HM_CLASSIFY_USER_PROMPT,
)
from utils.pos_keyphrase import extract_keyphrases
from utils.premise_indexer import number_sentences, pull_by_indices

MAX_CHAIN_DEPTH = 3
_RETRY_WAIT     = 30
_MAX_RETRIES    = 5
FALLBACK_LABEL  = "Neutral"
NOT_ANSWERABLE  = "NOT_ANSWERABLE"


# ── Retry helper ──────────────────────────────────────────────────────────────

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


# ── Pipeline class ────────────────────────────────────────────────────────────

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
        return _call_with_retry(llm.invoke, messages)

    # ── Stage 2a: Sentence Locator (reuses h_question prompts) ───────────────

    def stage2a_locate(
        self, numbered_premise: str, sub_question: str
    ) -> HLocateOutput | None:
        messages = [
            SystemMessage(content=H_LOCATE_SYSTEM_PROMPT),
            HumanMessage(content=H_LOCATE_USER_PROMPT.format(
                numbered_premise=numbered_premise, question=sub_question,
            )),
        ]
        llm = get_structured_llm(self.model, HLocateOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    # ── Stage 2b: Per-hop Answer Extractor ───────────────────────────────────

    def stage2b_answer(
        self, sub_question: str, sentences: list[str], context: str
    ) -> HMAnswerOutput | None:
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
        llm = get_structured_llm(self.model, HMAnswerOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    # ── Stage 3: Chain Classifier ─────────────────────────────────────────────

    def stage3_classify(
        self, hypothesis: str, chain_block: str
    ) -> HCompareOutput | None:
        messages = [
            SystemMessage(content=HM_CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=HM_CLASSIFY_USER_PROMPT.format(
                hypothesis=hypothesis, chain_block=chain_block,
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
                "hop_index":          hop_idx,
                "sub_question":       sub_q,
                "context_at_hop":     context,
                "located_indices":    [],
                "extracted_sentences": [],
                "answer_from_P":      NOT_ANSWERABLE,
                "answerable_flag":    False,
                "halted":             False,
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

        # Stage 3 — classify on answered hops (partial chain is fine)
        answered = [h for h in hops if h["answer_from_P"] != NOT_ANSWERABLE]
        final_label = FALLBACK_LABEL
        if not answered:
            warnings.append("No hop was answered; defaulting to Neutral.")
        else:
            chain_block = "\n\n".join(
                f"Hop {h['hop_index'] + 1}:\n"
                f"  Q: {h['sub_question']}\n"
                f"  A: {h['answer_from_P']}"
                for h in answered
            )
            try:
                clf = self.stage3_classify(hypothesis, chain_block)
                if clf:
                    final_label = clf.label
                else:
                    warnings.append("Stage 3 returned None; defaulting to Neutral.")
            except Exception as exc:
                logger.warning(f"Stage 3 failed: {exc}")
                warnings.append(f"Stage 3 error: {exc}")

        return self._make_result(sample, keyphrases, hops, final_label, warnings)

    @staticmethod
    def _make_result(
        sample:      dict,
        keyphrases:  list[str],
        hops:        list[dict],
        final_label: str,
        warnings:    list[str],
    ) -> dict:
        # Backward-compatible per_question_details (mirrors h_question shape)
        per_question_details = [
            {
                "question":              h["sub_question"],
                "located_indices":       h["located_indices"],
                "gold_evidence_indices": None,
                "extracted_sentences":   h["extracted_sentences"],
                "answer_from_P":         h["answer_from_P"],
                "answerable_flag":       h["answerable_flag"],
                "per_question_label":    "N/A",
            }
            for h in hops
        ]
        return {
            "id":                   sample.get("id"),
            "premise":              sample["premise"],
            "hypothesis":           sample["hypothesis"],
            "label":                sample["label"],
            "prediction":           final_label,
            "keyphrases":           keyphrases,
            "gen_questions":        [h["sub_question"] for h in hops],
            "per_question_details": per_question_details,
            "chain":                hops,
            "warnings":             warnings,
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
