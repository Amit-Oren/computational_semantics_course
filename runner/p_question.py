"""
P-Question Pipeline Runner — Premise Interrogation for NLI
==========================================================
Four-stage pipeline:
  Stage 1a: Premise → factual questions the premise answers          (LLM)
  Stage 1b: Score questions vs. hypothesis via ROUGE-L / BLEU;
            keep top-K highest-scoring questions                     (metric)
  Stage 1c: Two-step answer process per question                     (LLM × 2)
            Step 1 — Evidence Gathering: collect ALL relevant premise
                     sentences before forming any answer.
            Step 2 — Answer Synthesis: summarise gathered evidence
                     into one clean, final answer.
  Stage 2:  NLI classification from the extracted evidence           (LLM)

Ablation flags (set in config/config.py — one line each):
  P_QUESTION_ALIGNMENT_METRIC : "ROUGE_L" | "BLEU"
  P_QUESTION_STAGE2_MODE      : "concatenated" | "majority_vote"
"""

from __future__ import annotations

import time
from collections import Counter

from langchain_core.messages import SystemMessage, HumanMessage

from config.config import (
    DEFAULT_PARAMS,
    P_QUESTION_TOP_K,
    P_QUESTION_ALIGNMENT_METRIC,
    P_QUESTION_STAGE2_MODE,
    PQuestionListOutput,
    PEvidenceGatheringOutput,
    PAnswerOutput,
    PNLIOutput,
    get_structured_llm,
    logger,
)
from prompts.p_question import (
    P_QUESTION_SYSTEM_PROMPT,
    P_QUESTION_USER_PROMPT,
    P_GATHER_SYSTEM_PROMPT,
    P_GATHER_USER_PROMPT,
    P_ANSWER_SYSTEM_PROMPT,
    P_ANSWER_USER_PROMPT,
    P_NLI_SYSTEM_PROMPT,
    P_NLI_USER_PROMPT,
)

_RETRY_WAIT  = 30
_MAX_RETRIES = 5
FALLBACK_LABEL = "Neutral"
UNANSWERABLE   = "[UNANSWERABLE]"

_SPACY_NLP = None  # loaded once on first use


# ── Retry helper (mirrors q2_pipeline) ───────────────────────────────────────

def _call_with_retry(fn, *args, **kwargs):
    """Retry fn on capacity/rate-limit errors; re-raise anything else."""
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


# ── Alignment metrics ─────────────────────────────────────────────────────────

def _rouge_l_score(question: str, hypothesis: str) -> float:
    from rouge_score import rouge_scorer  # lazy import — not needed if BLEU is used
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(hypothesis, question)["rougeL"].fmeasure


def _bleu_score(question: str, hypothesis: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref  = hypothesis.lower().split()
    cand = question.lower().split()
    return sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method4)


def _align_score(question: str, hypothesis: str, metric: str) -> float:
    return _bleu_score(question, hypothesis) if metric == "BLEU" else _rouge_l_score(question, hypothesis)


# ── Pipeline class ────────────────────────────────────────────────────────────

class PQuestionPipeline:
    """Four-stage NLI pipeline: premise interrogation + evidence classification."""

    def __init__(self, model: str, params: dict):
        self.model             = model
        self.params            = params
        self.top_k             = P_QUESTION_TOP_K
        self.alignment_metric  = P_QUESTION_ALIGNMENT_METRIC
        self.stage2_mode       = P_QUESTION_STAGE2_MODE

    # ── NER coverage (additive, purely metric-based) ──────────────────────────

    def _ner_coverage_questions(
        self, premise: str, existing_questions: list[str]
    ) -> list[str]:
        """Return gap-filling questions for entities not covered by existing_questions."""
        global _SPACY_NLP
        try:
            import spacy as _spacy
            if _SPACY_NLP is None:
                _SPACY_NLP = _spacy.load("en_core_web_sm")
        except Exception as exc:
            logger.warning(f"spaCy unavailable — NER coverage skipped: {exc}")
            return []

        doc = _SPACY_NLP(premise)
        covered_lower = " ".join(existing_questions).lower()
        seen_texts: set[str] = set()
        new_questions: list[str] = []

        for ent in doc.ents:
            text = ent.text.strip()
            if len(text) < 3:
                continue
            key = text.lower()
            if key in seen_texts:
                continue
            seen_texts.add(key)
            if key in covered_lower:
                continue

            label = ent.label_
            if label in ("PERCENT", "CARDINAL", "QUANTITY"):
                q = f"What was the {text} figure mentioned in the premise?"
            elif label in ("DATE", "TIME"):
                q = f"What happened in/at {text} according to the premise?"
            else:
                q = f"What does the premise say about {text}?"

            new_questions.append(q)

        return new_questions

    # ── Stage 1a ─────────────────────────────────────────────────────────────

    def stage1a_generate_questions(self, premise: str) -> PQuestionListOutput | None:
        """Ask the LLM what factual questions the premise directly answers."""
        messages = [
            SystemMessage(content=P_QUESTION_SYSTEM_PROMPT),
            HumanMessage(content=P_QUESTION_USER_PROMPT.format(premise=premise)),
        ]
        llm = get_structured_llm(self.model, PQuestionListOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    # ── Stage 1b ─────────────────────────────────────────────────────────────

    def stage1b_align_questions(
        self,
        questions: list[str],
        hypothesis: str,
    ) -> tuple[list[dict], list[str]]:
        """Score every question against the hypothesis; return top-K + warnings."""
        warnings: list[str] = []
        scored = [
            {"question": q, "score": _align_score(q, hypothesis, self.alignment_metric)}
            for q in questions
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_k = scored[: self.top_k]

        if top_k and all(item["score"] == 0.0 for item in top_k):
            warnings.append(
                f"All {self.alignment_metric} scores are zero; "
                f"keeping top-{self.top_k} by generation order."
            )
        return top_k, warnings

    # ── Stage 1c ─────────────────────────────────────────────────────────────

    def _gather_evidence(
        self, question: str, premise: str
    ) -> PEvidenceGatheringOutput | None:
        """Step 1: scan the full premise and collect every relevant sentence."""
        messages = [
            SystemMessage(content=P_GATHER_SYSTEM_PROMPT),
            HumanMessage(content=P_GATHER_USER_PROMPT.format(
                premise=premise, question=question,
            )),
        ]
        llm = get_structured_llm(self.model, PEvidenceGatheringOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    def _synthesize_answer(
        self, question: str, evidence_sentences: list[str]
    ) -> PAnswerOutput | None:
        """Step 2: synthesise gathered evidence into one final answer."""
        evidence_block = "\n".join(f"- {s}" for s in evidence_sentences)
        messages = [
            SystemMessage(content=P_ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=P_ANSWER_USER_PROMPT.format(
                question=question, evidence_block=evidence_block,
            )),
        ]
        llm = get_structured_llm(self.model, PAnswerOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    def _extract_one_answer(
        self, question: str, premise: str
    ) -> tuple[PAnswerOutput | None, list[str]]:
        """Two-step Stage 1c for one question.

        Returns (PAnswerOutput | None, evidence_sentences).
        Evidence sentences are always returned so they can be logged,
        even when synthesis fails.
        """
        # Step 1 — evidence gathering
        gather_out = None
        try:
            gather_out = self._gather_evidence(question, premise)
        except Exception as exc:
            logger.warning(f"Stage 1c Step 1 (gather) failed for '{question[:60]}': {exc}")

        if gather_out is None or not gather_out.has_evidence or not gather_out.evidence_sentences:
            return None, []

        evidence_sentences = gather_out.evidence_sentences

        # Step 2 — answer synthesis
        answer_out = None
        try:
            answer_out = self._synthesize_answer(question, evidence_sentences)
        except Exception as exc:
            logger.warning(f"Stage 1c Step 2 (synthesize) failed for '{question[:60]}': {exc}")

        return answer_out, evidence_sentences

    def stage1c_extract_answers(
        self,
        aligned_items: list[dict],
        premise: str,
    ) -> list[dict]:
        """Return {question, score, evidence_sentences, answer, unanswerable} per question."""
        results = []
        for item in aligned_items:
            answer_out, evidence_sentences = self._extract_one_answer(item["question"], premise)

            if answer_out is None:
                results.append({
                    "question":          item["question"],
                    "score":             item["score"],
                    "evidence_sentences": evidence_sentences,
                    "answer":            UNANSWERABLE,
                    "unanswerable":      True,
                })
            else:
                # Belt-and-suspenders: honour both the boolean flag and the text
                is_unans = answer_out.unanswerable or (UNANSWERABLE in answer_out.answer.upper())
                results.append({
                    "question":          item["question"],
                    "score":             item["score"],
                    "evidence_sentences": evidence_sentences,
                    "answer":            answer_out.answer,
                    "unanswerable":      is_unans,
                })
        return results

    # ── Stage 2 ──────────────────────────────────────────────────────────────

    def _nli_call(self, evidence: str, hypothesis: str) -> PNLIOutput | None:
        messages = [
            SystemMessage(content=P_NLI_SYSTEM_PROMPT),
            HumanMessage(content=P_NLI_USER_PROMPT.format(
                evidence=evidence, hypothesis=hypothesis,
            )),
        ]
        llm = get_structured_llm(self.model, PNLIOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    def stage2_classify(
        self,
        answer_items: list[dict],
        hypothesis: str,
    ) -> tuple[str, str, list[str]]:
        """Return (predicted_label, evidence_string, warnings)."""
        warnings: list[str] = []
        answerable = [a for a in answer_items if not a["unanswerable"]]

        if not answerable:
            warnings.append(
                "All answers are [UNANSWERABLE]; skipping Stage 2 — defaulting to Neutral."
            )
            return FALLBACK_LABEL, "", warnings

        if self.stage2_mode == "majority_vote":
            evidence_string = " | ".join(a["answer"] for a in answerable)
            labels: list[str] = []
            for a in answerable:
                try:
                    out = self._nli_call(a["answer"], hypothesis)
                    labels.append(out.label if out else FALLBACK_LABEL)
                except Exception as exc:
                    logger.warning(f"Stage 2 majority_vote call failed: {exc}")
                    labels.append(FALLBACK_LABEL)
            counts   = Counter(labels)
            top_cnt  = max(counts.values())
            winners  = [lbl for lbl, cnt in counts.items() if cnt == top_cnt]
            # Tie → Neutral
            predicted = FALLBACK_LABEL if len(winners) > 1 else winners[0]
            return predicted, evidence_string, warnings

        # concatenated (default)
        evidence_string = " ".join(a["answer"] for a in answerable)
        try:
            out = self._nli_call(evidence_string, hypothesis)
            if out is None:
                warnings.append("Stage 2 returned no output; defaulting to Neutral.")
                return FALLBACK_LABEL, evidence_string, warnings
            return out.label, evidence_string, warnings
        except Exception as exc:
            warnings.append(f"Stage 2 call failed ({exc}); defaulting to Neutral.")
            return FALLBACK_LABEL, evidence_string, warnings

    # ── Full sample ───────────────────────────────────────────────────────────

    def run_sample(self, sample: dict) -> dict | None:
        premise    = sample["premise"]
        hypothesis = sample["hypothesis"]
        warnings:  list[str] = []

        # Stage 1a
        q_output = None
        try:
            q_output = self.stage1a_generate_questions(premise)
        except Exception as exc:
            logger.warning(f"Stage 1a failed: {exc}")

        if q_output is None or not q_output.questions:
            warnings.append("Stage 1a returned no questions; defaulting to Neutral.")
            return self._make_result(sample, [], [], [], "", FALLBACK_LABEL, warnings)

        questions = q_output.questions

        ner_questions = self._ner_coverage_questions(premise, questions)
        if ner_questions:
            logger.info(f"NER coverage added {len(ner_questions)} question(s): {ner_questions}")
            questions = questions + ner_questions

        # Stage 1b
        aligned_items, align_warnings = self.stage1b_align_questions(questions, hypothesis)
        warnings.extend(align_warnings)

        # Stage 1c
        answer_items = self.stage1c_extract_answers(aligned_items, premise)

        # Stage 2
        predicted_label, evidence_string, stage2_warnings = self.stage2_classify(
            answer_items, hypothesis,
        )
        warnings.extend(stage2_warnings)

        return self._make_result(
            sample, questions, aligned_items, answer_items,
            evidence_string, predicted_label, warnings,
        )

    @staticmethod
    def _make_result(
        sample:           dict,
        stage1a_questions: list[str],
        stage1b_aligned:   list[dict],
        stage1c_answers:   list[dict],
        stage2_evidence:   str,
        predicted_label:   str,
        warnings:          list[str],
    ) -> dict:
        return {
            "id":                sample.get("id"),
            "premise":           sample["premise"],
            "hypothesis":        sample["hypothesis"],
            "label":             sample["label"],
            "prediction":        predicted_label,
            "stage1a_questions": stage1a_questions,
            "stage1b_aligned":   stage1b_aligned,
            "stage1c_answers":   stage1c_answers,
            "stage2_evidence":   stage2_evidence,
            "stage2_mode":       P_QUESTION_STAGE2_MODE,
            "alignment_metric":  P_QUESTION_ALIGNMENT_METRIC,
            "warnings":          warnings,
        }


# ── Public entry point (matches runner contract used by main.py) ──────────────

def run(samples: list[dict], model: str, params: dict = DEFAULT_PARAMS) -> list[dict]:
    pipeline = PQuestionPipeline(model, params)

    logger.info("=" * 60)
    logger.info("Experiment        : p_question")
    logger.info(f"Model             : {model}")
    logger.info(f"Temperature       : {params.get('temperature')}")
    logger.info(f"Max tokens        : {params.get('max_tokens')}")
    logger.info(f"Alignment metric  : {pipeline.alignment_metric}")
    logger.info(f"Stage 2 mode      : {pipeline.stage2_mode}")
    logger.info(f"Top-K questions   : {pipeline.top_k}")
    logger.info(f"Samples           : {len(samples)}")
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
