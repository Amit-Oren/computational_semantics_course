"""
Bridge-Question Pipeline Prompts — Both-Texts Bridging for NLI
================================================================
Stage 1 (Bridge Question Generator): Premise + Hypothesis (both texts) →
                                      2-4 sub-questions, at least one of
                                      which explicitly bridges/compares a
                                      premise fact to a hypothesis claim.

Evidence-finding (Locator + Answer Extractor) and final classification are
shared components — see utils/locator_extractor.py and
prompts/shared_classifier.py. Only question generation is method-specific
here.

Design goal: p_question (premise-blind) and h_question (hypothesis-blind)
each only see one text, so neither can generate a question that explicitly
connects a specific premise fact to a specific hypothesis claim — that gap
was the confirmed root cause of p_question's Neutral-collapse failures
(11/17 non-Neutral samples: the decisive comparison was never generated).
bridge_question sees both texts together so it can ask that question
directly. Bridging/comparison framing follows HotpotQA (Yang et al., EMNLP
2018) and DecompRC (Min et al., ACL 2019); the "don't answer H directly in
Stage 1" safeguard follows DecompRC's observation that decomposition models
often shortcut multi-hop questions into a single-hop guess.
"""

# ─── Stage 1: Bridge Question Generator ──────────────────────────────────────

BRIDGE_QUESTION_GEN_SYSTEM_PROMPT = """\
You generate BRIDGING questions for a Natural Language Inference task.

You are given a PREMISE (a long passage) and a HYPOTHESIS (a short claim). Your job is \
to generate 2-4 sub-questions that, when answered using ONLY the premise, will let a \
separate classifier decide whether the hypothesis is Entailed, Contradicted, or \
Neutral.

Step 1 — Identify the hypothesis's CORE CLAIM: the one specific thing it asserts (an \
action, a comparison, a future plan, a causal link, a quantity). Not the general topic \
of the sentence — the precise claim being made.

Step 2 — Generate 2-4 sub-questions:
- At least ONE question MUST be a bridging or comparison question that tests the CORE \
CLAIM itself, not merely an adjacent or related topic. Reuse the hypothesis's own key \
words for the claim (the specific action/comparison/quantity/timeframe it asserts) \
inside the question. A question about the same general subject that does not test the \
specific claim does NOT count as bridging, even if it shares entities with the \
hypothesis.
- Do NOT answer the hypothesis yourself. Only generate the questions. The answers come \
later, from the premise.
- Keep exact names, numbers, and qualifiers from the premise and hypothesis. Never \
replace them with generic words ("someone", "a place").
- Each question targets exactly one checkable fact or one comparison.

  ✗ Wrong (adjacent topic, misses the core claim):
      H: "Clothing companies are planning to offer financial services in the future."
      Core claim: a FUTURE PLAN to offer financial services (not current services,
      not the financial-services industry in general).
      Bad "bridge" Q: "What services do clothing companies currently offer?"
      (tests present-day services, never touches any future financial-services plan)

  ✓ Correct (tests the core claim directly):
      Bridge Q: "Does the premise mention any plan by clothing companies to offer
      financial services in the future?"

Output JSON only:
{"questions": ["q1", "q2", ...], "bridge_indices": [i, ...]}
where bridge_indices lists which of the questions (0-based) are the bridging/comparison \
ones.\
"""

BRIDGE_QUESTION_GEN_USER_PROMPT = """\
PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}

Generate 2-4 sub-questions, at least one bridging or comparison question, that will \
verify the hypothesis against the premise.\
"""


# ─── Stage 1 retry: forces a bridging question when none was produced ───────

BRIDGE_QUESTION_RETRY_USER_PROMPT = """\
PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}

Your previous questions did not include any bridging or comparison question that tests \
the hypothesis's CORE CLAIM (its specific action/comparison/plan/quantity — not just its \
general topic). Generate 2-4 sub-questions again, and this time make sure AT LEAST ONE \
question reuses the hypothesis's own key words for its specific claim and tests that \
claim directly (e.g. "Does the premise's figure for X match what the hypothesis claims \
for X?"). A question about the same subject that misses the specific claim does not count.\
"""
