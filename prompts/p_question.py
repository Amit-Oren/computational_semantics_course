"""
P-Question Pipeline Prompts — Premise Interrogation for NLI
===========================================================
Stage 1a has TWO interchangeable question-generation prompts, selected via
the `generation` flag on the runner (see runner/p_question.py), so the new
method can be compared against the old one rather than replacing it outright:

  "decomposition" (new) — Premise (blind to H) → one question per atomic
                           fact and per relation/comparison the premise
                           expresses. FActScore-style atomic facts (Min et
                           al., EMNLP 2023) + DAE-style relation verification
                           (Goyal & Durrett, 2020-2021), so coverage of the
                           premise's facts and relations is guaranteed by
                           construction — free-form generation was observed
                           to skip the specific relation/comparison a
                           hypothesis depends on (e.g. "X's skill in A
                           surpassed skill in B") even while covering
                           everything else salient in the premise.
  "freeform" (old/baseline) — Premise → up to 15 breadth-first wh-questions,
                           no explicit fact/relation distinction (every
                           question is tagged type="fact" downstream, so
                           n_relation is always 0 for this mode — that's the
                           structural difference the comparison is testing).

Stage 1b (Selection):          Questions × Hypothesis → relevance scores,
                                keep top-K (see utils/question_selectors.py;
                                "topk" or "mmr" selection, two scorers).

Evidence-finding (Locator + Answer Extractor) and final classification are
shared components — see utils/locator_extractor.py and
prompts/shared_classifier.py.

Design goal: ground the NLI decision in premise content by first surfacing
what the premise explicitly knows, then checking whether that knowledge
supports, refutes, or is silent about the hypothesis.
"""

# ─── Stage 1a (new): Atomic-Fact + Relation Decomposition ───────────────────

P_QUESTION_SYSTEM_PROMPT = """\
You decompose a PREMISE into checkable units for a fact-verification task.

Step 1 — Extract ATOMIC FACTS: break the premise into the smallest standalone 
factual statements. Each atomic fact contains exactly ONE piece of information 
(one number, date, event, or attribute). Include facts that are implied as
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


# ─── Stage 1a (seeded): Anchor-Guided Question Generator ────────────────────
# Used when a seeder (pos / svo / srl) is active. Takes premise + seeds
# produced by the seeder, generates one targeted question per anchor.

P_QUESTION_SEEDED_SYSTEM_PROMPT = """\
You are a Question Generator for Natural Language Inference.

Given a PREMISE and SEMANTIC ANCHORS (keyphrases, relations, or role structures \
extracted from the premise), generate one targeted question per anchor.

Each question must:
  1. Directly ask about the information the anchor identifies.
  2. Be answerable solely from the PREMISE — no world knowledge needed.
  3. Include full identifying context (names, locations, dates) in the question.
  4. NOT contain its own answer.
  5. Be a well-formed wh-question ending with (?).

If two anchors refer to the same fact, merge them into one question.
Skip anchors that are too vague to form a meaningful question.

Output format — JSON only, no extra text:
{"questions": ["question_1", "question_2", ...]}\
"""

P_QUESTION_SEEDED_USER_PROMPT = """\
Premise: "{premise}"

Semantic anchors:
{seeds}

Generate one question per anchor asking about that concept or relation in the premise.\
"""


# ─── Stage 1a (new): Atomic-Fact + Relation Decomposition — FEW-SHOT ────────

P_QUESTION_FEW_SHOT_SYSTEM_PROMPT = """\
You decompose a PREMISE into checkable units for a fact-verification task.

Step 1 — Extract ATOMIC FACTS: break the premise into the smallest standalone
factual statements. Each atomic fact contains exactly ONE piece of information
(one number, date, event, or attribute). Include facts that are implied as
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
{"questions": [{"q": "question_1", "type": "fact"}, {"q": "question_2", "type": "relation"}, ...]}

EXAMPLES:

Premise: "The government has decided to pay compensation to the tune of Rs. 1 lakh to the family members of those who are killed in railway accidents."
{"questions": [{"q": "What has the government decided to pay?", "type": "fact"}, {"q": "What is the amount of compensation the government will pay?", "type": "fact"}, {"q": "Who will receive the government compensation?", "type": "fact"}, {"q": "What condition triggers the compensation payment?", "type": "fact"}, {"q": "What event causes the government's obligation to pay compensation?", "type": "relation"}]}

Premise: "The first ever low-cost tablet with a three-hour battery which works only with Wi-Fi for accessing the Internet will be a game changer for India's rural population, who are not able to afford the high prices of existing tablets and even the cheaper netbooks."
{"questions": [{"q": "What is the battery life of the tablet described in the premise?", "type": "fact"}, {"q": "What type of internet connectivity does the tablet use?", "type": "fact"}, {"q": "What segment of the population is the tablet targeting?", "type": "fact"}, {"q": "What existing products can India's rural population not afford?", "type": "fact"}, {"q": "Why does the premise characterize the tablet as a 'game changer' for India's rural population?", "type": "relation"}]}

Premise: "The local citizens group submitted a memorandum to the civic authority for allowing them to convert the vacant plot in the locality into a garden at their own cost."
{"questions": [{"q": "Who submitted the memorandum?", "type": "fact"}, {"q": "To whom was the memorandum submitted?", "type": "fact"}, {"q": "What did the local citizens group request in their memorandum?", "type": "fact"}, {"q": "Who would bear the cost of converting the vacant plot?", "type": "fact"}, {"q": "What motivated the citizens group to submit the memorandum to the civic authority?", "type": "relation"}]}\
"""

# ─── Stage 1a (seeded): Anchor-Guided Question Generator — FEW-SHOT ─────────

P_QUESTION_SEEDED_FEW_SHOT_SYSTEM_PROMPT = """\
You are a Question Generator for Natural Language Inference.

Given a PREMISE and SEMANTIC ANCHORS (keyphrases, relations, or role structures \
extracted from the premise), generate one targeted question per anchor.

Each question must:
  1. Directly ask about the information the anchor identifies.
  2. Be answerable solely from the PREMISE — no world knowledge needed.
  3. Include full identifying context (names, locations, dates) in the question.
  4. NOT contain its own answer.
  5. Be a well-formed wh-question ending with (?).

If two anchors refer to the same fact, merge them into one question.
Skip anchors that are too vague to form a meaningful question.

Output format — JSON only, no extra text:
{"questions": ["question_1", "question_2", ...]}

EXAMPLES:

Premise: "The government has decided to pay compensation to the tune of Rs. 1 lakh to the family members of those who are killed in railway accidents."
Semantic anchors:
- government compensation
- Rs. 1 lakh
- railway accidents
- family members
{"questions": ["What compensation has the government decided to pay?", "What is the amount of the government's compensation?", "What event triggers the compensation according to the premise?", "Who are the recipients of the government compensation?"]}

Premise: "The first ever low-cost tablet with a three-hour battery which works only with Wi-Fi for accessing the Internet will be a game changer for India's rural population, who are not able to afford the high prices of existing tablets and even the cheaper netbooks."
Semantic anchors:
- low-cost tablet
- three-hour battery
- Wi-Fi internet
- India's rural population
- existing tablets
{"questions": ["What type of tablet does the premise describe?", "What is the battery life of the tablet?", "What internet access method does the tablet use?", "What population group is the tablet intended for?", "What existing products can India's rural population not afford?"]}

Premise: "The local citizens group submitted a memorandum to the civic authority for allowing them to convert the vacant plot in the locality into a garden at their own cost."
Semantic anchors:
- local citizens group
- civic authority
- vacant plot
- garden conversion
- own cost
{"questions": ["What did the local citizens group submit?", "To whom did the citizens group submit their memorandum?", "What does the citizens group want to do with the vacant plot?", "Who would fund the conversion of the plot into a garden?"]}\
"""

# ─── Stage 1a (old/baseline): Free-Form Question Generator ──────────────────
# Restored verbatim from before the decomposition rewrite, for direct
# comparison via `--generation freeform`.

P_QUESTION_FREEFORM_SYSTEM_PROMPT = """\
You are a Factual Question Extractor for Natural Language Inference.

Given a PREMISE, generate up to 15 precise wh-questions covering the facts \
it contains. Prioritise breadth over exhaustiveness.

Each question must:
  1. Be answerable solely from the premise.
  2. Target exactly one fact per question.
  3. Be a well-formed wh-question ending with (?).
  4. NOT contain its own answer.
  5. Include full identifying context — names, locations, groups, and time
     periods must appear in the question, not just in the answer.
     (e.g. "What percentage of EUROPEAN women participated..." not
           "What percentage of women participated...")
  6. Cover ALL claim types in the premise, including:
       - Quantitative facts (numbers, percentages, dates, counts)
       - Qualitative and evaluative claims (were there problems? what was the
         overall effect? what does the text characterize as X?)
       - Causal and relational claims (what caused X? what resulted from Y?)
       - Presupposed facts (facts implied but not the main point of a sentence)
Output format — JSON only, no extra text:
{"questions": ["question_1", "question_2", ...]}\
"""

P_QUESTION_FREEFORM_USER_PROMPT = """\
Premise: "{premise}"

Generate up to 15 factual questions this premise answers.\
"""
