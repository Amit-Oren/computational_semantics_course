"""
H-Question Pipeline Prompts — Hypothesis Interrogation for NLI
==============================================================
Stage 1 (Question Generator): H + keyphrases → 1-2 probe questions.

Evidence-finding (Locator + Answer Extractor) and final classification are
shared components — see utils/locator_extractor.py and
prompts/shared_classifier.py. Only question generation is method-specific here.

Design goal: force the model to ground every claim in the premise text BEFORE
deciding the label, eliminating hypothesis-only bias on long premises (~500 words).
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
