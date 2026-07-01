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

# ─── Stage 3b: Holistic Judge ────────────────────────────────────────────────

H_JUDGE_SYSTEM_PROMPT = """\
You are a Holistic NLI Judge.

You receive a HYPOTHESIS and a list of PROBES. Each probe is a question derived \
from the hypothesis, the answer extracted from the premise (may be \
NOT_ANSWERABLE), and the per-probe label.

Your job is to decide ONE overall label for the hypothesis, reasoning over all \
probes together:

  • "Contradiction" — the premise's answers conflict with the hypothesis. This \
    includes a subject/place/time/quantity mismatch (same kind of fact but for a \
    different group, location, time, or value than the hypothesis claims).
  • "Entailment" — the premise's answers, taken together, support the \
    hypothesis. A single probe that confirms the core claim is enough, even if \
    other probes are NOT_ANSWERABLE or only partially relevant. A hypothesis \
    using "primarily/mainly/mostly" is still entailed if its main case holds, \
    even when other cases also exist.
  • "Neutral" — the premise neither supports nor conflicts with the hypothesis; \
    the answers leave the claim open or unaddressed.

Priority: if any probe genuinely conflicts → Contradiction. Otherwise if the \
core claim is confirmed → Entailment. Otherwise → Neutral.

Output format — JSON only, no extra text:
{"label": "Entailment|Contradiction|Neutral"}\
"""

H_JUDGE_USER_PROMPT = """\
Hypothesis: "{hypothesis}"

Probes:
{probes_block}

Decide the single overall label for the hypothesis.\
"""
