"""
P-Question Pipeline Runner — Premise Interrogation for NLI
==========================================================
Four-stage pipeline:
  Stage 1a: Premise → questions, via one of two interchangeable generation
            methods (LLM, blind to H) — see prompts/p_question.py
  Stage 1b: Score all questions against H with the selected scorer, then pick
            the final top-K set with the selected selection mode
            (utils/question_selectors.py)
  Stage 1c: locate_and_answer per selected question                  (shared, LLM x2 each)
  Stage 2:  classify_evidence over the surviving (question, answer)
            pairs                                                    (shared, LLM)

Stage 1a has two swappable generation methods (`generation` flag), compared
side-by-side rather than one replacing the other:
  "decomposition" (new) — decomposes the premise into atomic facts AND
               relations/comparisons (FActScore-style atomic decomposition +
               DAE-style relation verification), guaranteeing coverage of
               the premise's facts and relations by construction — free-form
               generation was observed to miss the specific relation a
               hypothesis depended on even while covering everything else
               salient in the premise. Can produce 20-40 candidates per
               premise, so the combined (decomposed + NER-coverage) pool is
               capped, keeping all relation-type questions first
               (scarce, high-value) and filling the rest with fact-type up
               to the cap — see `_cap_questions`.
  "freeform" (old/baseline) — up to 15 breadth-first wh-questions, no
               fact/relation distinction (n_relation is always 0).

Stage 1b has two orthogonal, swappable knobs:
  `selector`:  "rouge_l" (baseline, word-overlap) or "embedding"
               (Sentence-BERT cosine similarity) — how relevance is scored.
  `selection`: "topk" (plain highest-K) or "mmr" (Maximal Marginal Relevance
               — balances relevance against redundancy so the kept set
               covers distinct facts, not just the single most-relevant one).
Stage 1c locate_and_answer and Stage 2 classify_evidence are identical
regardless of which Stage 1a/1b combo is active; see
utils/question_selectors.py.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from config.config import (
    DEFAULT_PARAMS,
    P_QUESTION_GENERATION,
    P_QUESTION_TOP_K,
    P_QUESTION_SELECTOR,
    P_QUESTION_SELECTION,
    P_QUESTION_MMR_LAMBDA,
    P_QUESTION_MAX_QUESTIONS,
    PQuestionListOutput,
    PQuestionFreeformOutput,
    NLIOutput,
    get_structured_llm,
    logger,
)
from prompts.p_question import (
    P_QUESTION_SYSTEM_PROMPT,
    P_QUESTION_FEW_SHOT_SYSTEM_PROMPT,
    P_QUESTION_USER_PROMPT,
    P_QUESTION_FREEFORM_SYSTEM_PROMPT,
    P_QUESTION_FREEFORM_USER_PROMPT,
    P_QUESTION_SEEDED_SYSTEM_PROMPT,
    P_QUESTION_SEEDED_FEW_SHOT_SYSTEM_PROMPT,
    P_QUESTION_SEEDED_USER_PROMPT,
)
from prompts.zero_shot import SYSTEM_PROMPT as ZS_SYSTEM, USER_PROMPT as ZS_USER
from prompts.shared_classifier import classify_evidence
from utils.aggregation import aggregate, AGGREGATION_MODES
from utils.locator_extractor import locate_and_answer
from utils.premise_indexer import number_sentences
from utils.question_selectors import score_questions
from utils.retry import call_with_retry
from utils.seeding import get_seeder, SEEDERS

METHOD = "p_question"
FALLBACK_LABEL = "Neutral"
GENERATION_MODES = ("decomposition", "freeform", "seeded")
DEFAULT_AGGREGATION = "aggregated"

_SPACY_NLP = None  # loaded once on first use


class PQuestionPipeline:
    """Swappable Stage 1a generation + Stage 1b scorer/selection + shared evidence-finding/classification."""

    def __init__(
        self,
        model: str,
        params: dict,
        generation: str = P_QUESTION_GENERATION,
        selector: str = P_QUESTION_SELECTOR,
        selection: str = P_QUESTION_SELECTION,
        top_k: int = P_QUESTION_TOP_K,
        mmr_lambda: float = P_QUESTION_MMR_LAMBDA,
        max_questions: int = P_QUESTION_MAX_QUESTIONS,
        seeder_name: str = "pos",
        aggregation: str = DEFAULT_AGGREGATION,
        few_shot: bool = False,
    ):
        if generation not in GENERATION_MODES:
            raise ValueError(f"Unknown generation mode '{generation}'; choose from {GENERATION_MODES}")
        if aggregation not in AGGREGATION_MODES:
            raise ValueError(f"Unknown aggregation '{aggregation}'; choose from {AGGREGATION_MODES}")
        if generation == "seeded" and seeder_name not in SEEDERS:
            raise ValueError(f"Unknown seeder '{seeder_name}'; choose from {sorted(SEEDERS)}")
        self.model         = model
        self.params        = params
        self.generation    = generation
        self.top_k         = top_k
        self.selector      = selector
        self.selection     = selection
        self.mmr_lambda    = mmr_lambda
        self.max_questions = max_questions
        self.seeder_name   = seeder_name
        self.aggregation   = aggregation
        self.few_shot      = few_shot
        self.seeder        = get_seeder(seeder_name, model=model, params=params) if generation == "seeded" else None

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

    def stage1a_generate_questions(self, premise: str) -> list[dict]:
        """Ask the LLM to generate Stage 1a questions using the active generation method.

        Returns a list of {"q", "type"} dicts uniformly — freeform and seeded
        questions are tagged type="fact", decomposition questions preserve
        the fact/relation distinction.
        """
        if self.generation == "freeform":
            messages = [
                SystemMessage(content=P_QUESTION_FREEFORM_SYSTEM_PROMPT),
                HumanMessage(content=P_QUESTION_FREEFORM_USER_PROMPT.format(premise=premise)),
            ]
            llm = get_structured_llm(self.model, PQuestionFreeformOutput, self.params)
            out = call_with_retry(llm.invoke, messages)
            if out is None:
                return []
            return [{"q": q, "type": "fact"} for q in out.questions]

        if self.generation == "seeded":
            seeds = self.seeder.seed(premise)
            if not seeds:
                logger.warning("Seeder returned no seeds; falling back to freeform.")
                messages = [
                    SystemMessage(content=P_QUESTION_FREEFORM_SYSTEM_PROMPT),
                    HumanMessage(content=P_QUESTION_FREEFORM_USER_PROMPT.format(premise=premise)),
                ]
                llm = get_structured_llm(self.model, PQuestionFreeformOutput, self.params)
                out = call_with_retry(llm.invoke, messages)
                if out is None:
                    return []
                return [{"q": q, "type": "fact"} for q in out.questions]

            seeds_str = "\n".join(f"- {s}" for s in seeds)
            seeded_sys = (
                P_QUESTION_SEEDED_FEW_SHOT_SYSTEM_PROMPT if self.few_shot
                else P_QUESTION_SEEDED_SYSTEM_PROMPT
            )
            messages = [
                SystemMessage(content=seeded_sys),
                HumanMessage(content=P_QUESTION_SEEDED_USER_PROMPT.format(
                    premise=premise, seeds=seeds_str,
                )),
            ]
            llm = get_structured_llm(self.model, PQuestionFreeformOutput, self.params)
            out = call_with_retry(llm.invoke, messages)
            if out is None:
                return []
            return [{"q": q, "type": "fact"} for q in out.questions]

        # decomposition (default)
        decomp_sys = P_QUESTION_FEW_SHOT_SYSTEM_PROMPT if self.few_shot else P_QUESTION_SYSTEM_PROMPT
        messages = [
            SystemMessage(content=decomp_sys),
            HumanMessage(content=P_QUESTION_USER_PROMPT.format(premise=premise)),
        ]
        llm = get_structured_llm(self.model, PQuestionListOutput, self.params)
        out = call_with_retry(llm.invoke, messages)
        if out is None:
            return []
        return [{"q": item.q, "type": item.type} for item in out.questions]

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

        # Stage 1a — question generation (premise-blind), method per self.generation
        decomposed: list[dict] = []
        try:
            decomposed = self.stage1a_generate_questions(premise)
        except Exception as exc:
            logger.warning(f"Stage 1a failed: {exc}")

        if not decomposed:
            warnings.append("Stage 1a returned no questions; falling back to zero-shot.")
            prediction = self._zero_shot_fallback(sample, warnings)
            return self._make_result(sample, [], [], prediction, "", warnings, 0, 0)

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

        # Stage 1b removed — all generated questions go directly to locate_and_answer
        selected = [{"question": q, "type": type_map.get(q, "fact")} for q in questions]

        # Stage 1c — shared locate_and_answer per question
        indexed_sentences, numbered_premise = number_sentences(premise)
        qa_pairs = []
        for item in selected:
            result = locate_and_answer(
                self.model, self.params, item["question"],
                indexed_sentences=indexed_sentences, numbered_premise=numbered_premise,
            )
            result["type"] = item["type"]
            if result["answerable"]:
                qa_pairs.append(result)

        # Stage 2 — aggregate over surviving Q/A pairs
        pred_reasoning = ""
        if not qa_pairs:
            warnings.append("All questions unanswerable; falling back to zero-shot.")
            prediction = self._zero_shot_fallback(sample, warnings)
        else:
            prediction, pred_reasoning = aggregate(self.aggregation, self.model, self.params, qa_pairs, hypothesis)

        return self._make_result(
            sample, decomposed, qa_pairs, prediction, pred_reasoning, warnings, n_fact, n_relation,
        )

    def _make_result(
        self,
        sample:         dict,
        all_questions:  list[dict],
        qa_pairs:       list[dict],
        prediction:     str,
        pred_reasoning: str,
        warnings:       list[str],
        n_fact:         int,
        n_relation:     int,
    ) -> dict:
        return {
            "id":                   sample.get("id"),
            "premise":              sample["premise"],
            "hypothesis":           sample["hypothesis"],
            "gold_label":           sample["label"],
            "prediction":           prediction,
            "prediction_reasoning": pred_reasoning,
            "qa_pairs":             qa_pairs,
            "method":               METHOD,
            "generation":           self.generation,
            "seeder":               self.seeder_name,
            "aggregation":          self.aggregation,
            "few_shot":             self.few_shot,
            "all_questions":        all_questions,
            "n_fact":               n_fact,
            "n_relation":           n_relation,
            "warnings":             warnings,
        }


# ── Public entry point (matches runner contract used by main.py) ──────────────

def run(
    samples: list[dict],
    model: str,
    params: dict = DEFAULT_PARAMS,
    generation: str = P_QUESTION_GENERATION,
    selector: str = P_QUESTION_SELECTOR,
    selection: str = P_QUESTION_SELECTION,
    top_k: int = P_QUESTION_TOP_K,
    mmr_lambda: float = P_QUESTION_MMR_LAMBDA,
    max_questions: int = P_QUESTION_MAX_QUESTIONS,
    seeder_name: str = "pos",
    aggregation: str = DEFAULT_AGGREGATION,
    few_shot: bool = False,
) -> list[dict]:
    pipeline = PQuestionPipeline(
        model, params, generation=generation, selector=selector, selection=selection,
        top_k=top_k, mmr_lambda=mmr_lambda, max_questions=max_questions,
        seeder_name=seeder_name, aggregation=aggregation, few_shot=few_shot,
    )

    logger.info("=" * 60)
    logger.info("Experiment        : p_question")
    logger.info(f"Model             : {model}")
    logger.info(f"Temperature       : {params.get('temperature')}")
    logger.info(f"Max tokens        : {params.get('max_tokens')}")
    logger.info(f"Generation        : {pipeline.generation}")
    if pipeline.generation == "seeded":
        logger.info(f"Seeder            : {pipeline.seeder_name}")
    logger.info(f"Aggregation       : {pipeline.aggregation}")
    logger.info(f"Few-shot          : {pipeline.few_shot}")
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
