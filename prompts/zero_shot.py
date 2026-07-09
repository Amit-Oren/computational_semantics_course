SYSTEM_PROMPT = """You are solving a Natural Language Inference task.

Given a PREMISE and a HYPOTHESIS, classify the relationship as exactly one of:
- Entailment: the hypothesis must be true based on the premise.
- Contradiction: the hypothesis must be false based on the premise.
- Neutral: the hypothesis may be true or false; the premise does not provide enough information.

Output format — JSON only, no extra text:
{"label": "Entailment|Contradiction|Neutral", "explanation": "<one-sentence justification>"}"""

USER_PROMPT = """
PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}"""
