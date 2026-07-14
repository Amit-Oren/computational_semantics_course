"""Shared NLI Classifier — single call over collected evidence.

Final classification stage shared by q2_pipeline, p_question, h_question,
h_multihop, and bridge_question. Question-generation and evidence-gathering
differ per method; this call is identical everywhere: one evidence block +
the hypothesis → one label. No per-question voting.

Two entry points, same prompt/schema/retry machinery underneath (`_classify`):
  classify_evidence()     — evidence block is (question, answer) pairs.
  classify_raw_evidence() — evidence block is plain located sentences, no
                             question framing. Used by retrieve_then_classify
                             to isolate the Locator's contribution from
                             question generation — the classifier prompt
                             itself doesn't care whether "EVIDENCE" is
                             Q/A-framed or raw sentences.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from config.config import ClassifyOutput, get_structured_llm, logger
from utils.retry import call_with_retry

FALLBACK_LABEL = "Neutral"

CLASSIFIER_SYSTEM_PROMPT = """\
You are a Strict NLI Classifier.

You receive EVIDENCE (question-answer pairs extracted from a premise) and a
HYPOTHESIS. Determine the logical relationship between them.

EVIDENCE SUFFICIENCY: judge each piece of evidence on its own merit first. If
ANY single piece of evidence is clearly decisive on its own, that is enough —
decide from it. Do not let additional evidence that is merely background,
tangential, or inconclusive water down or "average away" a piece of evidence
that is independently clear. A strong signal does not become weaker because
weaker evidence sits next to it in the same block.

Apply these rules in strict priority order:

1. CONTRADICTION — the evidence makes the hypothesis impossible or highly unlikely.
   Includes: a different value for the same entity/fact, a direct negation, or
   conditions that rule out the hypothesis.

2. ENTAILMENT — the evidence directly supports or strongly implies the hypothesis.
   Word-for-word match is not required if the evidence makes the hypothesis a
   reasonable, necessary conclusion. Presuppositions and set-membership facts count.

   MODALITY RULE: if the hypothesis uses "may/might/could" language, treat the
   evidence as entailing it when the evidence shows the event is POSSIBLE — full
   confirmation is not required for hedged/modal hypotheses.

3. NEUTRAL — only if the evidence genuinely does not address the hypothesis at all.
   Do not default to Neutral when evidence is relevant but indirect.

   SCOPE GUARD: partial or topic-specific evidence must NOT be treated as
   confirming "overall / always / same / identical" scope claims in the hypothesis.
   A partial match stays Neutral (or Contradiction, if it directly conflicts)
   unless the evidence itself covers the full scope claimed.

Output format — JSON only, no extra text:
{"label": "Entailment|Contradiction|Neutral"}\
"""

CLASSIFIER_USER_PROMPT = """\
EVIDENCE:
{evidence_block}

HYPOTHESIS: "{hypothesis}"

Classify the relationship between the evidence and the hypothesis.\
"""


def build_evidence_block(qa_pairs: list[dict]) -> str:
    """Render surviving (question, answer) pairs as one evidence block, keeping
    each question and its answer together."""
    return "\n\n".join(
        f"Q{i}: {pair['question']}\nA{i}: {pair['answer']}"
        for i, pair in enumerate(qa_pairs, start=1)
    )


def build_raw_evidence_block(sentences: list[str]) -> str:
    """Render raw located premise sentences as one evidence block — no
    question framing, since no question was generated."""
    return "\n".join(f"- {s}" for s in sentences)


def _classify(model: str, params: dict, evidence_block: str, hypothesis: str) -> str:
    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=CLASSIFIER_USER_PROMPT.format(
            evidence_block=evidence_block, hypothesis=hypothesis,
        )),
    ]
    llm = get_structured_llm(model, ClassifyOutput, params)
    try:
        out = call_with_retry(llm.invoke, messages)
    except Exception as exc:
        logger.warning(f"classifier call failed: {exc}")
        return FALLBACK_LABEL
    return out.label if out else FALLBACK_LABEL


def classify_evidence(model: str, params: dict, qa_pairs: list[dict], hypothesis: str) -> str:
    """Classify hypothesis against collected (question, answer) evidence pairs.

    `qa_pairs` must already be filtered to answerable pairs only. Returns
    "Neutral" directly (no LLM call) when qa_pairs is empty.
    """
    if not qa_pairs:
        return FALLBACK_LABEL
    return _classify(model, params, build_evidence_block(qa_pairs), hypothesis)


def classify_raw_evidence(model: str, params: dict, sentences: list[str], hypothesis: str) -> str:
    """Classify hypothesis against raw located premise sentences (no
    question/answer framing — no question was generated for this method).

    Returns "Neutral" directly (no LLM call) when sentences is empty.
    """
    if not sentences:
        return FALLBACK_LABEL
    return _classify(model, params, build_raw_evidence_block(sentences), hypothesis)
