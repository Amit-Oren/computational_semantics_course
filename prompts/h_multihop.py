"""
H-Multihop Pipeline Prompts — Atomic Sub-Question Chaining for NLI
===================================================================
Stage 1  (Decomposition Planner): H + keyphrases → ordered 2-3 sub-questions.
Stage 2a (Sentence Locator):      Numbered premise + sub-question → indices.
                                   (reuses H_LOCATE_* from prompts/h_question.py)
Stage 2b (Per-hop Answerer):      Extracted sentences + prior context → answer.
Stage 3  (Chain Classifier):      Full evidence chain → holistic NLI label.

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


# ─── Stage 3: Chain Classifier ───────────────────────────────────────────────

HM_CLASSIFY_SYSTEM_PROMPT = """\
You are a Chain-NLI Judge.

You receive a HYPOTHESIS and an EVIDENCE CHAIN — an ordered list of sub-questions \
and their answers extracted from the premise. The chain MAY BE PARTIAL: it can \
stop early if a later sub-question was not answerable. Judge using the hops that \
WERE answered — never default to Neutral just because the chain is incomplete.

A single answered hop is enough to decide when it settles the hypothesis:
  • If any answered hop conflicts with the hypothesis, label Contradiction even \
if a later hop is unanswered. (E.g. H "this was his first offence"; an answered \
hop says he offended before → Contradiction. H "police rang his mother"; an \
answered hop says he is an orphan / has no mother → Contradiction.)
  • If the answered hops fully confirm the hypothesis, label Entailment even if \
a later confirmatory hop is missing.

Decide ONE overall label:

  • "Contradiction" — any chain answer conflicts with what the hypothesis claims \
(wrong entity, wrong quantity, wrong direction, wrong location, or wrong time). \
A subject/place/time/quantity mismatch counts even if the same kind of fact exists.
  • "Entailment" — the chain answers taken together confirm the hypothesis. \
Partial, topic-specific, or single-example evidence does NOT entail a claim that \
uses "overall", "same", "always", or "identical" scope; label Entailment only \
when the answers cover the FULL scope the hypothesis claims.
  • "Neutral" — the chain does not confirm or conflict; evidence is absent, \
off-topic, or covers only part of what the hypothesis asserts.

Priority: any genuine conflict → Contradiction. Full-scope confirmation → \
Entailment. Otherwise → Neutral.

Output format — JSON only, no extra text:
{"label": "Entailment|Contradiction|Neutral"}\
"""

HM_CLASSIFY_USER_PROMPT = """\
Hypothesis: "{hypothesis}"

Evidence chain (may be partial — judge from the answered hops):
{chain_block}

Decide the single overall label.\
"""


# ─── Fix 1: REQUIRED runner change ───────────────────────────────────────────
# In your h_multihop runner, the chain currently auto-labels Neutral whenever a
# hop halts. Remove that shortcut and ALWAYS call the classifier on whatever was
# answered. Roughly:
#
#   # OLD (delete this):
#   # if any(h["halted"] for h in chain):
#   #     prediction = "Neutral"
#   # else:
#   #     prediction = classify_chain(hypothesis, chain)
#
#   # NEW:
#   answered = [h for h in chain if h["answer_from_P"] != "NOT_ANSWERABLE"]
#   if not answered:                       # nothing at all was answered
#       prediction = "Neutral"
#   else:
#       prediction = classify_chain(hypothesis, answered)   # judge partial chain
#
# Keep logging the full chain (halted hops included) for traceability; only the
# label decision changes.