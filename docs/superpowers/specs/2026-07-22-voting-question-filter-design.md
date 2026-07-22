# Relation-priority question filter for p_question voting mode

## Problem

`p_question`'s `voting` aggregation classifies each generated question's
(question, answer) pair independently, then majority-votes the labels
(`utils/aggregation.py::_voting`). Decomposition-based question generation
produces ~17-21 questions per sample on average (up to the existing 30 cap),
and only a small minority are actually decisive for the hypothesis — most are
isolated atomic facts (a date, a percentage, a name) that, judged completely
alone, correctly vote Neutral. Majority vote then gets swamped: e.g. 25
correctly-Neutral atomic-fact votes outnumber 2-3 genuinely decisive votes.
This is not a prompt bug (the classifier prompt already discourages
defaulting to Neutral) — it's a mismatch between decomposition's output
granularity and per-question majority voting.

Measured impact: `voting` scores 24-30% accuracy vs. `aggregated`'s 54-58%
on the same n=50 dev-set slice, same model, same JSON-repair-fixed code.

## Goal

Reduce the number of low-signal votes reaching the majority-vote step,
specifically for `voting` mode, without touching `aggregated`/
`sequential_cot` (which already benefit from seeing the full unfiltered
question set).

## Design

### New function: `utils/question_selectors.py::select_for_voting`

```python
def select_for_voting(
    questions: list[dict],   # [{"q": str, "type": "fact"|"relation"}, ...]
    hypothesis: str,
    cap: int = 10,
) -> list[dict]
```

Logic:
1. Split `questions` into `relations` and `facts` by the existing `type`
   field (set at Stage 1a generation time — no extra computation).
2. If `len(relations) > cap`: rank relations by ROUGE-L score against the
   hypothesis (reusing `_score_rouge_l`, already in this module), keep the
   top `cap`.
3. Otherwise: keep **all** relations, then fill the remaining
   `cap - len(relations)` slots with the top-ranked `fact` questions by
   ROUGE-L.
4. Return the combined list (length <= `cap`).

Rationale for relation-priority: relation-type questions (comparisons,
causal links, evaluative/superlative claims, temporal ordering) require
connecting two or more pieces of information — the same shape of reasoning
a decisive vote needs. Fact-type questions are single atomic data points,
almost always correctly Neutral in isolation. This is a zero-cost signal:
the type tag already exists on every question from Stage 1a with no
additional model or scoring call.

Rationale for `cap=10`: real `n_relation` distribution across our n=50 runs
is 0-11 (median 3, average 4.3); 0/50 samples exceed 15, only 1/50 exceeds
10. A cap of 15 would still fill ~10-12 slots with fact-type questions
(relations diluted to ~20-25% of the vote); a cap of 10 leaves relations
~40-50% of the vote while still leaving real headroom (~6 slots) for cases
where the decisive evidence happens to be fact-type.

### Wiring: `runner/p_question.py::PQuestionPipeline.run_sample`

Current code (post Stage-1b-removal):
```python
# Stage 1b removed — all generated questions go directly to locate_and_answer
selected = [{"question": q, "type": type_map.get(q, "fact")} for q in questions]
```

Becomes conditional on aggregation mode:
```python
if self.aggregation == "voting":
    filtered = select_for_voting(decomposed, hypothesis, cap=10)
    questions = [d["q"] for d in filtered]
    type_map = {d["q"]: d["type"] for d in filtered}
selected = [{"question": q, "type": type_map.get(q, "fact")} for q in questions]
```

`aggregated` and `sequential_cot` are untouched — same unfiltered question
set as today.

### Error handling

None added beyond what exists. ROUGE-L scoring is a deterministic local
library call (no network, no LLM) — there is no transient-failure mode here
worth guarding against; a missing/broken `rouge_score` install is a real
setup error that should surface immediately, not be silently caught.

### Testing

New file `tests/test_question_selectors.py`, following the existing
plain-`test_*`-function convention in `tests/test_h_question_utils.py`
(runnable via pytest or directly):

- all relations kept when `len(relations) <= cap`
- facts fill remaining slots, ranked by ROUGE-L score
- relations themselves trimmed by ROUGE-L when `len(relations) > cap`
- empty input returns empty list, no crash
- `cap=0` returns empty list, no crash

End-to-end validation (manual, not a unit test): rerun
`p_question | voting | few_shot=False` and `p_question | voting | few_shot=True`
on the same n=50 dev-set slice used for the existing 30.0%/24.0% baselines,
compare average question count (expect ~21 -> <=10) and accuracy.
