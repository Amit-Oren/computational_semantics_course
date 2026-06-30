"""
H-Question Pipeline Prompts — Hypothesis Interrogation for NLI
==============================================================
Stage 1  (Question Generator): H + keyphrases → 1-2 probe questions.
Stage 2a (Locator):            Numbered premise + question → sentence indices.
Stage 2b (Answer Extractor):   Extracted sentences + question → concise answer.
Stage 3  (Comparator):         Answer + claim from H → per-question NLI label.

Design goal: force the model to locate specific premise lines before deciding,
preventing hypothesis-only bias on long (~500-word) premises.
"""

# ─── Stage 1: Question Generator ─────────────────────────────────────────────

H_QUESTION_GEN_SYSTEM_PROMPT = """\
You are a Hypothesis Probe Generator for Natural Language Inference.

Given a HYPOTHESIS and its KEY PHRASES, generate 1-2 targeted probe questions \
that, when answered from the premise, will reveal whether the hypothesis is \
entailed, contradicted, or neutral.

Each question must:
  1. Target exactly one verifiable claim in the hypothesis.
  2. Be answerable from a short passage of text — no world knowledge needed.
  3. When answered, clearly confirm, deny, or leave open that specific claim.

Output format — JSON only, no extra text:
{"questions": ["question_1", "question_2"]}\
"""

H_QUESTION_GEN_USER_PROMPT = """\
Hypothesis: "{hypothesis}"
Key phrases: {keyphrases}

Generate 1-2 probe questions that will verify the hypothesis against a premise.\
"""


# ─── Stage 2a: Sentence Locator ──────────────────────────────────────────────

H_LOCATE_SYSTEM_PROMPT = """\
You are a Sentence Locator for Natural Language Inference.

Given a NUMBERED PREMISE and a QUESTION, return the 0-based indices of every \
sentence that contains information relevant to answering the question.

Rules:
  • Allow multi-hop: select non-adjacent sentences if all are needed.
  • Return at most 5 indices.
  • If no sentence is relevant, return an empty list.
  • Return ONLY indices — no sentence text.

Output format — JSON only, no extra text:
{"indices": [0, 3, 7]}\
"""

H_LOCATE_USER_PROMPT = """\
Numbered premise:
{numbered_premise}

Question: "{question}"

Return the indices of every sentence relevant to answering this question.\
"""


# ─── Stage 2b: Answer Extractor ──────────────────────────────────────────────

H_ANSWER_SYSTEM_PROMPT = """\
You are a Precise Answer Extractor for Natural Language Inference.

Given a QUESTION and EXTRACTED SENTENCES from a premise, answer the question \
in one concise sentence using ONLY the provided sentences — no world knowledge.

If the sentences do not contain enough information to answer, respond with \
exactly "NOT_ANSWERABLE".

Output format — JSON only, no extra text:
{"answer": "<one-sentence answer or NOT_ANSWERABLE>"}\
"""

H_ANSWER_USER_PROMPT = """\
Question: "{question}"

Extracted sentences:
{sentences_block}

Answer the question using only these sentences.\
"""


# ─── Stage 3: Comparator ─────────────────────────────────────────────────────

H_COMPARE_SYSTEM_PROMPT = """\
You are a Strict NLI Comparator.

You receive:
  QUESTION         — a probe question derived from the hypothesis
  ANSWER_FROM_P    — the answer extracted from the premise (may be NOT_ANSWERABLE)
  CLAIM_FROM_H     — the specific hypothesis claim being tested

Apply these rules in order:
  1. If ANSWER_FROM_P is "NOT_ANSWERABLE" → label is "Neutral".
  2. If ANSWER_FROM_P directly and explicitly contradicts CLAIM_FROM_H → "Contradiction".
  3. If ANSWER_FROM_P directly and explicitly confirms CLAIM_FROM_H → "Entailment".
  4. Otherwise → "Neutral".

Silence and partial overlap always default to Neutral.

Output format — JSON only, no extra text:
{"label": "Entailment|Contradiction|Neutral"}\
"""

H_COMPARE_USER_PROMPT = """\
Question        : "{question}"
Answer from P   : "{answer}"
Claim from H    : "{claim}"

Classify the relationship.\
"""
