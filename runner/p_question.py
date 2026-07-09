"""
P-Question Pipeline Runner — Premise Interrogation for NLI
==========================================================
Four-stage pipeline:
  Stage 1a: Premise → atomic-fact + relation decomposition, one question per
            unit (LLM, blind to H) — see prompts/p_question.py
  Stage 1b: Score all questions against H with the selected scorer, then pick
            the final top-K set with the selected selection mode
            (utils/question_selectors.py)
  Stage 1c: locate_and_answer per selected question                  (shared, LLM x2 each)
  Stage 2:  classify_evidence over the surviving (question, answer)
            pairs                                                    (shared, LLM)

Stage 1a decomposes the premise into atomic facts AND relations/comparisons
(FActScore-style atomic decomposition + DAE-style relation verification),
guaranteeing coverage of the premise's facts and relations by construction —
free-form generation was observed to miss the specific relation a hypothesis
depended on even while covering everything else salient in the premise. This
can produce 20-40 candidates per premise, well above the old free-form cap,
so the combined (decomposed + NER-coverage) pool is capped, keeping all
relation-type questions first (scarce, high-value) and filling the rest with
fact-type up to the cap — see `_cap_questions`.

Stage 1b has two orthogonal, swappable knobs:
  `selector`:  "rouge_l" (baseline, word-overlap), "embedding" (Sentence-BERT
               cosine similarity), or "llm_relevance" (LLM judges
               relevance-to-verdict, 0-5) — how relevance is scored.
  `selection`: "topk" (plain highest-K) or "mmr" (Maximal Marginal Relevance
               — balances relevance against redundancy so the kept set
               covers distinct facts, not just the single most-relevant one).
               With a larger decomposed candidate pool, MMR's coverage
               behavior finally has real redundancy to prune.
Stage 1c locate_and_answer and Stage 2 classify_evidence are identical
regardless of which Stage 1b combo is active; see utils/question_selectors.py.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from config.config import (
    DEFAULT_PARAMS,
    P_QUESTION_TOP_K,
    P_QUESTION_SELECTOR,
    P_QUESTION_SELECTION,
    P_QUESTION_MMR_LAMBDA,
    P_QUESTION_MAX_QUESTIONS,
    PQuestionListOutput,
    get_structured_llm,
    logger,
)
from prompts.p_question import (
    P_QUESTION_SYSTEM_PROMPT,
    P_QUESTION_USER_PROMPT,
)
from prompts.shared_classifier import classify_evidence
from utils.locator_extractor import locate_and_answer
from utils.premise_indexer import number_sentences
from utils.question_selectors import score_questions
from utils.retry import call_with_retry

METHOD = "p_question"
FALLBACK_LABEL = "Neutral"

_SPACY_NLP = None  # loaded once on first use


class PQuestionPipeline:
    """Premise decomposition + swappable Stage 1b scorer/selection + shared evidence-finding/classification."""

    def __init__(
        self,
        model: str,
        params: dict,
        selector: str = P_QUESTION_SELECTOR,
        selection: str = P_QUESTION_SELECTION,
        top_k: int = P_QUESTION_TOP_K,
        mmr_lambda: float = P_QUESTION_MMR_LAMBDA,
        max_questions: int = P_QUESTION_MAX_QUESTIONS,
    ):
        self.model         = model
        self.params        = params
        self.top_k         = top_k
        self.selector      = selector
        self.selection     = selection
        self.mmr_lambda    = mmr_lambda
        self.max_questions = max_questions

    # ── NER coverage (additive, purely metric-based) ──────────────────────────

    def _ner_coverage_questions(
        self, premise: str, existing_questions: list[str]
    ) -> list[str]:
        """Return gap-filling questions for entities not covered by existing_questions."""
        if len(premise.split()) < 300:
            return []
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
        """Ask the LLM to decompose the premise into atomic-fact + relation questions."""
        messages = [
            SystemMessage(content=P_QUESTION_SYSTEM_PROMPT),
            HumanMessage(content=P_QUESTION_USER_PROMPT.format(premise=premise)),
        ]
        llm = get_structured_llm(self.model, PQuestionListOutput, self.params)
        return call_with_retry(llm.invoke, messages)

    @staticmethod
    def _cap_questions(items: list[dict], cap: int) -> list[dict]:
        """Keep all relation-type questions first (scarce, high-value), then
        fill the remainder with fact-type up to the cap."""
        if len(items) <= cap:
            return items
        relations = [d for d in items if d["type"] == "relation"]
        facts     = [d for d in items if d["type"] != "relation"]
        kept = relations[:cap]
        remaining = cap - len(kept)
        if remaining > 0:
            kept += facts[:remaining]
        return kept

    # ── Stage 1b ─────────────────────────────────────────────────────────────

    def stage1b_select_questions(
        self,
        questions: list[str],
        hypothesis: str,
    ) -> list[dict]:
        """Score every question against the hypothesis with the active
        selector, then pick the final set with the active selection mode;
        return {"question", "relevance"} dicts."""
        scored = score_questions(
            questions, hypothesis, self.selector, self.top_k,
            selection=self.selection, mmr_lambda=self.mmr_lambda,
            model=self.model, params=self.params,
        )
        return [{"question": q, "relevance": relevance} for q, relevance in scored]

    # ── Full sample ───────────────────────────────────────────────────────────

    def run_sample(self, sample: dict) -> dict:
        premise    = sample["premise"]
        hypothesis = sample["hypothesis"]
        warnings:  list[str] = []

        # Stage 1a — atomic-fact + relation decomposition (premise-blind)
        q_output = None
        try:
            q_output = self.stage1a_generate_questions(premise)
        except Exception as exc:
            logger.warning(f"Stage 1a failed: {exc}")

        if q_output is None or not q_output.questions:
            warnings.append("Stage 1a returned no questions; defaulting to Neutral.")
            return self._make_result(sample, [], [], [], FALLBACK_LABEL, warnings, 0, 0)

        decomposed = [{"q": item.q, "type": item.type} for item in q_output.questions]

        ner_questions = self._ner_coverage_questions(premise, [d["q"] for d in decomposed])
        if ner_questions:
            logger.info(f"NER coverage added {len(ner_questions)} question(s): {ner_questions}")
            decomposed += [{"q": q, "type": "fact"} for q in ner_questions]

        n_generated = len(decomposed)
        decomposed = self._cap_questions(decomposed, self.max_questions)
        if len(decomposed) < n_generated:
            warnings.append(
                f"Capped {n_generated} generated questions down to {len(decomposed)} "
                f"(relations kept first)."
            )

        questions = [d["q"] for d in decomposed]
        type_map  = {d["q"]: d["type"] for d in decomposed}
        n_fact     = sum(1 for d in decomposed if d["type"] == "fact")
        n_relation = sum(1 for d in decomposed if d["type"] == "relation")

        # Stage 1b — swappable scorer + selection mode
        selected = self.stage1b_select_questions(questions, hypothesis)
        for item in selected:
            item["type"] = type_map.get(item["question"], "fact")

        # Stage 1c — shared locate_and_answer per selected question
        indexed_sentences, numbered_premise = number_sentences(premise)
        qa_pairs = []
        for item in selected:
            result = locate_and_answer(
                self.model, self.params, item["question"],
                indexed_sentences=indexed_sentences, numbered_premise=numbered_premise,
            )
            result["relevance"] = item["relevance"]
            result["type"]      = item["type"]
            if result["answerable"]:
                qa_pairs.append(result)

        # Stage 2 — shared classifier
        if not qa_pairs:
            warnings.append("All questions unanswerable; defaulting to Neutral.")
            prediction = FALLBACK_LABEL
        else:
            prediction = classify_evidence(self.model, self.params, qa_pairs, hypothesis)

        return self._make_result(
            sample, decomposed, selected, qa_pairs, prediction, warnings, n_fact, n_relation,
        )

    def _make_result(
        self,
        sample:        dict,
        all_questions: list[dict],
        selected:      list[dict],
        qa_pairs:      list[dict],
        prediction:    str,
        warnings:      list[str],
        n_fact:        int,
        n_relation:    int,
    ) -> dict:
        return {
            "id":            sample.get("id"),
            "premise":       sample["premise"],
            "hypothesis":    sample["hypothesis"],
            "gold_label":    sample["label"],
            "prediction":    prediction,
            "qa_pairs":      qa_pairs,
            "method":        METHOD,
            "selector":      self.selector,
            "selection":     self.selection,
            "mmr_lambda":    self.mmr_lambda,
            "all_questions": all_questions,
            "n_fact":        n_fact,
            "n_relation":    n_relation,
            "selected":      selected,
            "warnings":      warnings,
        }


# ── Public entry point (matches runner contract used by main.py) ──────────────

def run(
    samples: list[dict],
    model: str,
    params: dict = DEFAULT_PARAMS,
    selector: str = P_QUESTION_SELECTOR,
    selection: str = P_QUESTION_SELECTION,
    top_k: int = P_QUESTION_TOP_K,
    mmr_lambda: float = P_QUESTION_MMR_LAMBDA,
    max_questions: int = P_QUESTION_MAX_QUESTIONS,
) -> list[dict]:
    pipeline = PQuestionPipeline(
        model, params, selector=selector, selection=selection,
        top_k=top_k, mmr_lambda=mmr_lambda, max_questions=max_questions,
    )

    logger.info("=" * 60)
    logger.info("Experiment        : p_question")
    logger.info(f"Model             : {model}")
    logger.info(f"Temperature       : {params.get('temperature')}")
    logger.info(f"Max tokens        : {params.get('max_tokens')}")
    logger.info(f"Selector          : {pipeline.selector}")
    logger.info(f"Selection         : {pipeline.selection}")
    if pipeline.selection == "mmr":
        logger.info(f"MMR lambda        : {pipeline.mmr_lambda}")
    logger.info(f"Top-K questions   : {pipeline.top_k}")
    logger.info(f"Max questions     : {pipeline.max_questions}")
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

        results.append(result)
        warn_str = f" | warnings={result['warnings']}" if result["warnings"] else ""
        logger.info(
            f"[{i+1}/{len(samples)}] id={sample_id} "
            f"| gold={result['gold_label']} | pred={result['prediction']} "
            f"| n_fact={result['n_fact']} n_relation={result['n_relation']}{warn_str}"
        )

    if results:
        correct  = sum(r["gold_label"] == r["prediction"] for r in results)
        accuracy = correct / len(results)
        logger.info("=" * 60)
        logger.info(f"Processed : {len(results)} samples  |  Skipped: {skipped}")
        logger.info(f"Accuracy  : {correct}/{len(results)} = {accuracy:.4f}")
        logger.info("=" * 60)

    return results
