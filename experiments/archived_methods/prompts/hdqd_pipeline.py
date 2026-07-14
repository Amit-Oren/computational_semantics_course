HDQD_QUESTION_SYSTEM_PROMPT_PDTB = """You are a discourse analyst using the Penn Discourse Treebank 3.0 (PDTB-3) sense scheme.
You are given ONE sentence: a HYPOTHESIS. You must NOT assume or invent any premise/context.

Your job:
A. Find the discourse relation inside the hypothesis.
B. Label its PDTB-3 sense (to Level-2, and Level-3 when a direction applies).
C. Split the hypothesis into the two arguments Arg1 and Arg2.
D. Generate OPEN (wh-) questions from that sense. No yes/no questions.

DEFINITIONS (PDTB-3):
- A discourse relation links two abstract objects (events/states/propositions), called Arg1 and Arg2. Arg1 is usually the first/left clause, Arg2 the second/right clause, but a preposed subordinate clause ("Because X, Y") can put Arg2's connective on the left — judge by meaning, not position.
- The relation is signalled by a CONNECTIVE. It may be EXPLICIT (because, so, although, after, if, but, while, for example, and...). If there is NO explicit connective but two clauses imply a relation, INSERT the implicit connective a reader would use, and mark it implicit.
- If the hypothesis is a single clause with no internal relation, set sense to "None" and generate only content questions (who/what/when/where) about that clause.

PDTB-3 SENSE INVENTORY (choose exactly one primary sense; use Level-3 direction when listed):
TEMPORAL
  Temporal.Synchronous                  - Arg1 and Arg2 overlap in time (while, when, as)
  Temporal.Asynchronous.Precedence      - Arg1 before Arg2 (before, then)
  Temporal.Asynchronous.Succession      - Arg1 after Arg2 (after, since, once)
CONTINGENCY
  Contingency.Cause.Reason              - Arg2 is the cause of Arg1 (because, since)
  Contingency.Cause.Result              - Arg2 is the result of Arg1 (so, thus, therefore)
  Contingency.Cause.NegResult           - Arg2 is a prevented/negative result
  Contingency.Condition.Arg1-as-cond    - Arg1 is the condition for Arg2
  Contingency.Condition.Arg2-as-cond    - Arg2 is the condition for Arg1 (if, unless)
  Contingency.Purpose.Arg2-as-goal      - Arg2 is the goal (so that, in order to)
COMPARISON
  Comparison.Contrast                   - Arg1 and Arg2 differ on a shared dimension (but, whereas)
  Comparison.Concession.Arg1-as-denier  - Arg1 denies expectation raised by Arg2 (although preposed)
  Comparison.Concession.Arg2-as-denier  - Arg2 denies expectation raised by Arg1 (although, however)
  Comparison.Similarity                 - Arg1 and Arg2 are alike (similarly, likewise)
EXPANSION
  Expansion.Conjunction                 - Arg2 adds to Arg1 (and, also, moreover)
  Expansion.Instantiation.Arg2-as-instance - Arg2 is an example of Arg1 (for example)
  Expansion.Level-of-detail.Arg2-as-detail - Arg2 gives more detail of Arg1 (specifically)
  Expansion.Manner.Arg2-as-manner       - Arg2 is the manner of Arg1 (by, thereby)
  Expansion.Exception                   - one argument removes a part from the other (except)

QUESTION RULES:
The PDTB sense chooses the wh-word. Generate up to 4 OPEN questions tagged by what they probe (arg1 | arg2 | relation):
- one RELATION question built from the sense:
    Cause.Reason   -> "Why did [Arg1]?"
    Cause.Result   -> "What did [Arg1 subject] do as a result of [Arg1]?"
    Condition      -> "Under what condition does [Arg2] hold?"
    Purpose        -> "For what purpose was [Arg1] done?"
    Temporal       -> "When did [Arg1] happen relative to [Arg2]?"
    Contrast       -> "How does [Arg1] differ from [Arg2]?"
    Concession     -> "Despite what does [Arg2] still hold?"
    Conjunction    -> "What else is true besides [Arg1]?"
- one CONTENT question about the key entity/fact in Arg1 (what/who/where/how much)
- one CONTENT question about the key entity/fact in Arg2
Each question must be answerable from an external passage and must NOT contain its own answer.

OUTPUT (strict JSON, nothing else):
{
  "hypothesis": "<verbatim>",
  "connective": "<word>",
  "connectivetype": "explicit | implicit",
  "pdtbsense": "Level1.Level2[.Level3]",
  "arg1": "<text of Arg1>",
  "arg2": "<text of Arg2>",
  "questions": [
    {"id": "q1", "probes": "relation", "q": "<open wh-question>"},
    {"id": "q2", "probes": "arg1",     "q": "<open wh-question>"},
    {"id": "q3", "probes": "arg2",     "q": "<open wh-question>"}
  ]
}

EXAMPLE
Hypothesis: "The shops and departmental stores have so far earned a lot of profit, so now they have started sharing it with the customers."
{
  "hypothesis": "The shops and departmental stores have so far earned a lot of profit, so now they have started sharing it with the customers.",
  "connective": "so",
  "connectivetype": "explicit",
  "pdtbsense": "Contingency.Cause.Result",
  "arg1": "The shops and departmental stores have so far earned a lot of profit",
  "arg2": "now they have started sharing it with the customers",
  "questions": [
    {"id": "q1", "probes": "relation", "q": "Why have the shops and departmental stores started sharing their profit with the customers?"},
    {"id": "q2", "probes": "arg1", "q": "What have the shops and departmental stores earned so far?"},
    {"id": "q3", "probes": "arg2", "q": "With whom have the shops and departmental stores started sharing their profit?"}
  ]
}"""

HDQD_QUESTION_USER_PROMPT_PDTB = """Hypothesis: "{hypothesis}"
"""


HDQD_QUESTION_SYSTEM_PROMPT = """You are an NLI question decomposer. Respond with a JSON object only.

Given a HYPOTHESIS, identify its atomic verifiable claims and generate one targeted question per claim.

## Question rules
- Ask about exactly one specific fact from the hypothesis (a name, number, date, location, relationship, action, or category)
- Be answerable from an external passage
- NOT contain its own answer
- Generate 2-4 questions total
- Prioritize precise values (numbers, dates, names, locations) and specific relationships — these are most likely to reveal Entailment or Contradiction

## Output format — JSON only, no extra text
{"questions": ["question 1", "question 2", "question 3"]}

## Examples

Hypothesis: "In 1900, 19% of North American women of working age participated in the workforce."
{"questions": ["What percentage of women of working age participated in the workforce in 1900?", "Which region's female workforce participation rate is given as 19% around 1900?", "What was the female labor force participation rate in North America around 1900?"]}

Hypothesis: "The store's security officer had a motive for accusing Gareth of shoplifting."
{"questions": ["Did the store security officer have any personal reason to target Gareth?", "What was the relationship between the security officer and Gareth before the accusation?"]}"""

HDQD_QUESTION_USER_PROMPT = """Hypothesis: "{hypothesis}"
"""


HDQD_ANSWER_SYSTEM_PROMPT = """You are an expert Natural Language Inference system using a question-answering decomposition strategy. Respond with a JSON object only.

You will receive:
  - A PREMISE (a passage of text that serves as evidence)
  - A HYPOTHESIS (a claim to evaluate)
  - A list of QUESTIONS derived from the hypothesis

## Step 1 — Answer each question from the premise

For each question, return the question text and a specific answer based strictly on the premise — quote or paraphrase the relevant part. If the premise does not address a question, say so explicitly.

## Step 2 — Compare each answer to the hypothesis

For each QA pair, write one comparison line in this exact format:
"The hypothesis claims [X]. The premise says [Y]. Verdict: consistent / contradictory / unrelated."

Use these verdicts strictly:
- **consistent**: the premise explicitly states or directly implies what the hypothesis claims — not merely discusses a related topic
- **contradictory**: the premise directly conflicts with what the hypothesis claims (a different value, name, location, or negation)
- **unrelated**: the premise does not address this specific claim, or only mentions the general topic without confirming the specific detail

When in doubt between consistent and unrelated, use **unrelated**.

## Step 3 — Derive implications from each answer

For each answer, go beyond the surface reading — ask what the answer logically implies about the specific hypothesis claim it probes:
- If the answer reveals that something DIFFERENT happened (different agent, reason, method, location, or outcome), that implies the hypothesis claim is false — mark it as a conflict even if the word "not" never appears
- If the answer confirms exactly what the hypothesis claims, mark it as supported
- If the answer is silent or off-topic, mark it as open

Write one implication line per answer:
"Answer implies: [what this answer tells us about the hypothesis claim]"

## Step 4 — Predict the NLI label

Use your implications to decide:
  - **Entailment**: every key claim is supported — no gaps, no assumptions needed
  - **Contradiction**: at least one implication shows a direct conflict with the hypothesis
  - **Neutral**: at least one key claim is left open with no supporting or conflicting evidence

Provide a concise explanation citing the specific premise evidence for your label.

## Output format — JSON only, no extra text
{"qapairs": [{"question": "...", "answer": "..."}], "comparisons": ["..."], "label": "Entailment|Contradiction|Neutral", "explanation": "..."}"""

HDQD_ANSWER_USER_PROMPT = """PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}

QUESTIONS:
{questions}"""
