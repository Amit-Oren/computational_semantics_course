"""Stage 1b question-selection for p_question (ablation).

Two orthogonal knobs, used to pick which questions Stage 1c
(locate_and_answer) actually answers. Everything else in p_question —
premise-blind question generation in Stage 1a, the shared Locator/Extractor,
and the shared classify_evidence — is unaffected by either.

`selector` — how relevance to the hypothesis is scored:
  rouge_l        — ROUGE-L F-measure (baseline; word-overlap, no model load).
  embedding      — Sentence-BERT cosine similarity (semantic, no LLM call).
  llm_relevance  — LLM judges relevance-to-verdict on a 0-5 scale (one call
                   per question).

`selection` — how the final top-K set is chosen from those relevance scores:
  topk — plain highest-K by relevance (may keep several questions that all
         cover the same fact).
  mmr  — Maximal Marginal Relevance (Carbonell & Goldstein, SIGIR 1998):
         greedily balances relevance against redundancy with the
         already-selected set, so the kept set covers distinct facts
         instead of just the single most-relevant one. Redundancy is always
         measured via Sentence-BERT cosine similarity (reuses the same
         embedding model as the `embedding` selector), regardless of which
         `selector` scored relevance.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from config.config import LLMRelevanceOutput, get_structured_llm, logger
from utils.retry import call_with_retry

SELECTORS = ("rouge_l", "embedding", "llm_relevance")
SELECTION_MODES = ("topk", "mmr")

_EMBEDDING_MODEL = None  # lazy-loaded once, cached (mirrors the spaCy NER pattern)


# ─── Scorer 1: ROUGE-L (baseline) ────────────────────────────────────────────

def _score_rouge_l(question: str, hypothesis: str, **_) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(hypothesis, question)["rougeL"].fmeasure


# ─── Scorer 2: Sentence-BERT cosine similarity ───────────────────────────────

def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def _score_embedding(question: str, hypothesis: str, **_) -> float:
    from sentence_transformers import util
    model = _get_embedding_model()
    embeddings = model.encode([question, hypothesis], convert_to_tensor=True)
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())


# ─── Scorer 3: LLM relevance-to-verdict judge ────────────────────────────────

LLM_RELEVANCE_SYSTEM_PROMPT = """\
You judge whether a QUESTION is relevant to deciding a HYPOTHESIS.

Given a QUESTION (generated from a premise) and a HYPOTHESIS, rate how much \
knowing the answer to this question would help decide whether the hypothesis is \
True, False, or Unrelated to the premise.

Score 0-5:
  5 = answering this question would directly decide the hypothesis
  3 = answering this would give partial evidence toward the hypothesis
  0 = answering this is irrelevant to the hypothesis either way

Judge by MEANING and logical role, not by shared words. A question can be highly \
relevant even if it shares no words with the hypothesis (e.g. a question about \
"European" figures is highly relevant to a hypothesis about "North American" \
figures, because the answer decides a possible contradiction).

Output JSON only: {"score": <int 0-5>}\
"""

LLM_RELEVANCE_USER_PROMPT = """\
QUESTION: "{question}"
HYPOTHESIS: "{hypothesis}"

Rate how relevant this question is to deciding the hypothesis.\
"""


def _score_llm_relevance(question: str, hypothesis: str, model: str = None, params: dict = None, **_) -> float:
    messages = [
        SystemMessage(content=LLM_RELEVANCE_SYSTEM_PROMPT),
        HumanMessage(content=LLM_RELEVANCE_USER_PROMPT.format(
            question=question, hypothesis=hypothesis,
        )),
    ]
    llm = get_structured_llm(model, LLMRelevanceOutput, params)
    try:
        out = call_with_retry(llm.invoke, messages)
    except Exception as exc:
        logger.warning(f"llm_relevance scoring failed for '{question[:60]}': {exc}")
        return 0.0
    return float(out.score) if out else 0.0


_SCORER_FNS = {
    "rouge_l":       _score_rouge_l,
    "embedding":     _score_embedding,
    "llm_relevance": _score_llm_relevance,
}


_MAX_SCORING_WORKERS = 8  # only matters for llm_relevance — its calls are network I/O


def _raw_scores(
    questions: list[str], hypothesis: str, selector: str, model: str, params: dict
) -> dict[str, float]:
    scorer_fn = _SCORER_FNS[selector]

    if selector == "llm_relevance" and len(questions) > 1:
        # Each call is an independent network round-trip — run them
        # concurrently instead of one at a time (rouge_l/embedding score
        # locally and are already fast, so they stay sequential).
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=_MAX_SCORING_WORKERS) as pool:
            scores = list(pool.map(
                lambda q: scorer_fn(q, hypothesis, model=model, params=params),
                questions,
            ))
        return dict(zip(questions, scores))

    return {q: scorer_fn(q, hypothesis, model=model, params=params) for q in questions}


def _normalize_relevance(selector: str, raw_score: float) -> float:
    """Map every selector's raw score onto [0, 1] so relevance is comparable
    across selectors and usable directly in the MMR formula."""
    if selector == "llm_relevance":
        return max(0.0, min(1.0, raw_score / 5.0))
    return max(0.0, min(1.0, raw_score))  # ROUGE-L / cosine are already ~[0,1]


# ─── MMR: Maximal Marginal Relevance (Carbonell & Goldstein, SIGIR 1998) ─────

def _cosine(a, b) -> float:
    from sentence_transformers import util
    return float(util.cos_sim(a, b).item())


def _embed_cache(questions: list[str]) -> dict[str, object]:
    model = _get_embedding_model()
    embeddings = model.encode(questions, convert_to_tensor=True)
    return {q: embeddings[i] for i, q in enumerate(questions)}


def mmr_select(
    questions: list[str],
    rel_scores: dict[str, float],
    top_k: int = 3,
    lam: float = 0.7,
) -> list[str]:
    """Greedy MMR: MMR(q) = lam * Rel(q, H) - (1 - lam) * max_{s in S} Sim(q, s).

    rel_scores must already be normalized to [0, 1]. Redundancy is always
    measured via Sentence-BERT cosine similarity between candidate and
    already-selected questions, independent of which selector produced
    rel_scores. lam=1.0 zeroes the redundancy term, reducing exactly to
    top-K by relevance.
    """
    if not questions:
        return []
    if len(questions) <= top_k:
        return sorted(questions, key=lambda q: rel_scores[q], reverse=True)

    emb = _embed_cache(questions)
    candidates = list(questions)
    selected: list[str] = []

    first = max(candidates, key=lambda q: rel_scores[q])
    selected.append(first)
    candidates.remove(first)

    while candidates and len(selected) < top_k:
        def _mmr_score(q: str) -> float:
            redundancy = max(_cosine(emb[q], emb[s]) for s in selected)
            return lam * rel_scores[q] - (1 - lam) * redundancy
        nxt = max(candidates, key=_mmr_score)
        selected.append(nxt)
        candidates.remove(nxt)

    return selected


def score_questions(
    questions: list[str],
    hypothesis: str,
    selector: str,
    top_k: int = 3,
    *,
    selection: str = "topk",
    mmr_lambda: float = 0.7,
    model: str = None,
    params: dict = None,
) -> list[tuple[str, float]]:
    """Score every question against the hypothesis with the chosen selector,
    then pick the final top_k set with the chosen selection mode ("topk" or
    "mmr"). Returns (question, normalized_relevance) pairs for the selected
    set, relevance in [0, 1]."""
    if selector not in _SCORER_FNS:
        raise ValueError(f"Unknown selector '{selector}'; choose from {SELECTORS}")
    if selection not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode '{selection}'; choose from {SELECTION_MODES}")
    if not questions:
        return []

    raw = _raw_scores(questions, hypothesis, selector, model, params)
    rel = {q: _normalize_relevance(selector, s) for q, s in raw.items()}

    if selection == "mmr":
        selected_qs = mmr_select(questions, rel, top_k=top_k, lam=mmr_lambda)
    else:
        selected_qs = sorted(questions, key=lambda q: rel[q], reverse=True)[:top_k]

    return [(q, rel[q]) for q in selected_qs]
