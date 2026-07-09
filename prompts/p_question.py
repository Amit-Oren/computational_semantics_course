"""
P-Question Pipeline Prompts — Premise Interrogation for NLI
===========================================================
Stage 1a (Atomic-Fact + Relation Decomposition): Premise (blind to H) →
                                one question per atomic fact and per relation
                                /comparison the premise expresses.
Stage 1b (Selection):          Questions × Hypothesis → relevance scores,
                                keep top-K (see utils/question_selectors.py;
                                "topk" or "mmr" selection, three scorers).

Evidence-finding (Locator + Answer Extractor) and final classification are
shared components — see utils/locator_extractor.py and
prompts/shared_classifier.py.

Design goal: ground the NLI decision in premise content by first surfacing
what the premise explicitly knows, then checking whether that knowledge
supports, refutes, or is silent about the hypothesis. Stage 1a decomposes the
premise into its smallest checkable units (FActScore-style atomic facts;
Min et al., EMNLP 2023) AND the relations/comparisons between them (DAE-style
relation verification; Goyal & Durrett, 2020-2021), so coverage of the
premise's facts and relations is guaranteed by construction — free-form
generation was observed to skip the specific relation/comparison a
hypothesis depends on (e.g. "X's skill in A surpassed skill in B") even
while covering everything else salient in the premise.
"""

# ─── Stage 1a: Atomic-Fact + Relation Decomposition ──────────────────────────

P_QUESTION_SYSTEM_PROMPT = """\
You decompose a PREMISE into checkable units for a fact-verification task.

Step 1 — Extract ATOMIC FACTS: break the premise into the smallest standalone \
factual statements. Each atomic fact contains exactly ONE piece of information \
(one number, date, event, or attribute). Include facts that are implied as \
background, not only the main points.

Step 2 — Extract RELATIONS: identify how facts connect — causal links \
("what caused X?"), comparisons between two things ("which was greater, X or \
Y?"), evaluative claims ("what does the premise characterize as the \
strongest/main/most/least X?"), and temporal ordering ("what happened before \
Y?"). This step is critical — do not skip it even when the premise reads as a \
plain list of facts; look for every pair of facts the premise implicitly ranks, \
contrasts, or causally connects.

Step 3 — Turn each atomic fact and each relation into ONE question that the \
premise answers. Each question:
  - Targets exactly one atomic fact or one relation — never both.
  - Includes full identifying context — names, places, groups, and time \
periods must appear in the question itself, not be left implicit.
    (e.g. "What percentage of EUROPEAN women participated..." not
          "What percentage of women participated...")
  - Does NOT contain its own answer.
  - Is a well-formed wh-question ending with (?).

Label each question "fact" or "relation" accordingly.

Output format — JSON only, no extra text:
{"questions": [{"q": "question_1", "type": "fact"}, {"q": "question_2", "type": "relation"}, ...]}\
"""

P_QUESTION_USER_PROMPT = """\
Premise: "{premise}"

Decompose this premise into atomic-fact and relation/comparison questions.\
"""
