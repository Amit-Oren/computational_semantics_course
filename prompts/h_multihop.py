"""
H-Multihop Pipeline Prompts — Atomic Sub-Question Chaining for NLI
===================================================================
Stage 1  (Decomposition Planner): H + keyphrases → ordered 2-3 sub-questions.
Stage 2a (Sentence Locator):      Numbered premise + sub-question → indices.
                                   (reuses LOCATE_* from prompts/shared_answering.py)
Stage 2b (Per-hop Answerer):      Extracted sentences + prior context → answer.
                                   (kept local — needs the running context param,
                                   unlike the shared Answer Extractor)
Stage 3  (Chain Classifier):      Replaced by the shared classify_evidence() —
                                   see prompts/shared_classifier.py. The answered
                                   hops become its qa_pairs list.

CHANGELOG:
  Fix 2 — Decomposition keeps the exact named entities from the hypothesis
          (no replacing names with "someone"/"something").
  Fix 4 — If the hypothesis presupposes an entity/thing exists, the FIRST
          sub-question must check that existence in the premise.
  Fix 1 — A halted chain no longer auto-defaults to Neutral. All ANSWERED hops
          (including those before a halt) are passed to the classifier, which
          may return Contradiction/Entailment from a partial chain. See the
          runner change note at the bottom of this file.
"""

# ─── Stage 1: Decomposition Planner ──────────────────────────────────────────

HM_DECOMP_SYSTEM_PROMPT = """\
You are a Decomposition Planner for Natural Language Inference.

Given a HYPOTHESIS and its KEY PHRASES, produce 2-3 atomic sub-questions that, \
when answered in order from a premise, together verify the hypothesis.

Ordering rules:
  1. EXISTENCE FIRST. If the hypothesis assumes that a person, object, or thing \
exists (e.g. "his mother", "the company's CEO", "the second report"), the FIRST \
sub-question MUST check whether that entity actually exists in the premise \
(e.g. "Does Gareth have a mother?"). Only after existence is settled may later \
sub-questions ask what that entity did.
  2. If no such presupposed entity exists, the first sub-question must instead \
establish the most basic presupposed fact (a definition or threshold) that \
later sub-questions depend on.
  3. Each subsequent sub-question may explicitly refer to the answer of the \
previous one.
  4. Each sub-question must be atomic — target exactly one verifiable fact, \
no compound questions.
  5. Maximum 3 sub-questions.

Entity rules (IMPORTANT):
  • Keep the EXACT names and entities from the hypothesis. If the hypothesis \
says "Gareth", every sub-question must say "Gareth" — never replace a name with \
"someone", "a person", "something", or any other generic word.
  • Keep specific qualifiers from the hypothesis (places, dates, quantities, \
roles) in the sub-questions that test them. Do not broaden "North American \
women" to "women", or "in 1900" to "at some point".

Output format — JSON only, no extra text:
{"sub_questions": ["q1", "q2", "q3"]}\
"""

HM_DECOMP_USER_PROMPT = """\
Hypothesis: "{hypothesis}"
Key phrases: {keyphrases}

Produce 2-3 ordered atomic sub-questions. Keep the exact names and qualifiers \
from the hypothesis, and put any existence check first.\
"""


# ─── Stage 2b: Per-hop Answer Extractor ──────────────────────────────────────

HM_HOP_ANSWER_SYSTEM_PROMPT = """\
You are a Contextual Answer Extractor for Natural Language Inference.

You receive PRIOR CONTEXT (answers to earlier sub-questions in this chain; \
may be empty), a SUB-QUESTION to answer, and EXTRACTED SENTENCES from the premise.

Rules:
  1. Use ONLY the extracted sentences and the prior context — no outside knowledge.
  2. You may make AT MOST ONE small, obvious inference if it follows directly \
from the sentence text (e.g., "X is an orphan" → "X has no living parents"). \
Do not chain inferences.
  3. If the sentences are related to the sub-question but do not fully confirm \
the claim, still report what they DO say in one sentence — do not bail out.
  4. Return exactly "NOT_ANSWERABLE" ONLY when the sentences contain NO \
information related to the sub-question at all.

Output format — JSON only, no extra text:
{"answer": "<one-sentence answer or NOT_ANSWERABLE>", "answerable": true|false}\
"""

HM_HOP_ANSWER_USER_PROMPT = """\
Prior context:
{context_block}

Sub-question: "{sub_question}"

Extracted sentences:
{sentences_block}

Answer the sub-question using only these sentences and the prior context.\
"""