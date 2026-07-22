# ConTRoL NLI — Ablation Summary

Model: `qwen2.5-32b` · Dataset: ConTRoL test split (unless noted) · N=50 samples (unless noted)

**Important dating note:** the original table below (§ "Results (N=50, stale — pre-fix)") predates two significant fixes made 2026-07-21/22: a lab-server connectivity bug (wrong Tailscale IP + wrong URL path in `.env`, causing silent multi-minute hangs) and a JSON-parsing bug in `p_question`'s decomposition output (an unquoted enum value that silently discarded ~14% of samples to a zero-shot fallback). Those numbers should be treated as historical, not current — see § "Results (N=50, current code, 2026-07-22)" for verified re-runs.

---

## Pipelines

**zero_shot** — Direct NLI. One call: `(P, H) → label`. No reasoning steps.

**few_shot_cot** — Same but with 3 fixed demonstrations (one per label) in the prompt + chain-of-thought instruction. Forces the model to reason before labeling.

**retrieve_then_classify** — Control baseline, added after the original table below. No question generation at all: the Locator runs directly on the hypothesis itself as its query, and the classifier reads the raw located sentences (not a paraphrased "answer"). Isolates how much of any question-based method's gain comes from locating evidence alone vs. the questions themselves.

**h_question** — Interrogate H via P.
1. Extract seeds from H (keyphrases)
2. Generate probe questions from those seeds
3. Locate answers in P (Locator: is it answerable? what's the answer?)
4. Classify: given all Q/A pairs → NLI label

**p_question** — Decompose P, test against H.
1. Generate questions *about* P (from decomposed facts/relations or seeded keyphrases)
2. Locate answers in P
3. ~~Filter top-K by relevance to H~~ — **removed** in the current code for `aggregated`/`sequential_cot` (all generated questions go straight to Stage 1c unfiltered; see § below). Voting mode has its own separate, still-unresolved filter.
4. Classify: given Q/A pairs → NLI label

**bridge_question** — Questions that span both P and H.
1. Generate questions requiring both P and H to answer
2. Locate answers in P
3. Classify from Q/A pairs

---

## Ablation Axes

### Seeder — how seed phrases are extracted from the text
Used by **h_question** (seeds from H) and **p_question|seeded** (seeds from P).

- **`pos`** — NLTK POS tagger extracts noun phrases and named entities. Deterministic, fast, no LLM call.
- **`srl`** — LLM extracts PropBank-style semantic roles (agent, theme, time, location…). Richer than POS but depends on another LLM call.

### Generation — how questions are formed from P
Used by **p_question only**.

- **`decomposition`** — LLM reads P and breaks it into atomic **facts** and **relations** (comparisons, causal links, evaluative claims, temporal ordering), tagging each question accordingly.
- **`seeded`** — Keyphrases extracted from P by the seeder (pos or srl), then passed to the LLM as seeds. Two sub-configs: seeded|pos and seeded|srl.

### Aggregation — how Q/A pairs are combined into a final NLI label
Used by **h_question**, **p_question**, **bridge_question**.

- **`aggregated`** — All Q/A pairs passed together in a single classifier call.
- **`sequential_cot`** — Q/A pairs processed one at a time, running verdict updated after each.
- **`voting`** — Each Q/A pair classified independently (one call per pair); majority vote wins.

### Few-shot (added after the original table below)
Toggle on Stage 1's question-generation prompt for `h_question`/`p_question`/`bridge_question` — off (zero-shot instructions only) or on (3 worked examples appended). Not present as an axis in the original 20-config sweep below.

---

## Results (N=50, stale — pre-fix, kept for historical reference)

| Config | Acc | Macro F1 | Entail R | Contr R | Neutral R | N-fail |
|---|---|---|---|---|---|---|
| **h_question \| pos \| aggregated** | **0.780** | **0.760** | 0.800 | 0.688 | 0.889 | 9 |
| h_question \| pos \| voting | 0.740 | 0.728 | 0.680 | 0.750 | 0.889 | 10 |
| h_question \| srl \| aggregated | 0.740 | 0.720 | 0.760 | 0.688 | 0.778 | 11 |
| bridge_question \| aggregated | 0.720 | 0.698 | 0.720 | 0.688 | 0.778 | 11 |
| **few_shot_cot** *(baseline)* | **0.700** | **0.682** | 0.720 | 0.500 | 1.000 | 14 |
| h_question \| srl \| sequential_cot | 0.700 | 0.684 | 0.720 | 0.563 | 0.889 | 13 |
| h_question \| srl \| voting | 0.700 | 0.683 | 0.680 | 0.688 | 0.778 | 9 |
| **zero_shot** *(baseline)* | **0.640** | **0.618** | 0.680 | 0.375 | 1.000 | 18 |
| h_question \| pos \| sequential_cot | 0.640 | 0.647 | 0.560 | 0.625 | 0.889 | 17 |
| bridge_question \| voting | 0.620 | 0.604 | 0.640 | 0.500 | 0.778 | 16 |
| bridge_question \| sequential_cot | 0.580 | 0.580 | 0.520 | 0.563 | 0.778 | 17 |
| p_question \| decomposition \| aggregated | 0.460 | 0.467 | 0.400 | 0.375 | 0.778 | 24 |
| p_question \| seeded\|pos \| aggregated | 0.440 | 0.451 | 0.320 | 0.438 | 0.778 | 24 |
| p_question \| seeded\|srl \| aggregated | 0.400 | 0.413 | 0.280 | 0.375 | 0.778 | 26 |
| p_question \| decomposition \| sequential_cot | 0.320 | 0.329 | 0.160 | 0.313 | 0.778 | 29 |
| p_question \| seeded\|srl \| sequential_cot | 0.320 | 0.326 | 0.200 | 0.250 | 0.778 | 29 |
| p_question \| seeded\|srl \| voting | 0.280 | 0.275 | 0.080 | 0.250 | 0.889 | 34 |
| p_question \| decomposition \| voting | 0.260 | 0.237 | 0.040 | 0.188 | 1.000 | 36 |
| p_question \| seeded\|pos \| sequential_cot | 0.280 | 0.279 | 0.160 | 0.188 | 0.778 | 33 |
| p_question \| seeded\|pos \| voting | 0.240 | 0.218 | 0.080 | 0.125 | 0.889 | 37 |

*N-fail = samples where gold is Entailment/Contradiction but model predicted Neutral*

---

## Results (N=50, current code, 2026-07-22 — verified after both fixes)

Only `p_question` and `retrieve_then_classify` have been re-verified under current code so far; `h_question`/`bridge_question` still need fresh runs (using the stale numbers above as reference in the meantime).

| Config | Acc |
|---|---|
| **retrieve_then_classify** | **0.700** |
| p_question \| decomposition \| aggregated \| few-shot on | 0.580 |
| p_question \| decomposition \| aggregated \| few-shot off | 0.540 |
| p_question \| decomposition \| voting \| few-shot off (unfiltered) | 0.300 |
| p_question \| decomposition \| voting \| type-priority filter \| few-shot off | 0.300 |
| p_question \| decomposition \| voting \| few-shot on (unfiltered) | 0.240 |
| p_question \| decomposition \| voting \| ROUGE-L filter \| few-shot off | 0.240 |
| p_question \| decomposition \| voting \| type-priority filter \| few-shot on | 0.220 |
| p_question \| decomposition \| voting \| ROUGE-L filter \| few-shot on | 0.220 |

---

## Key Findings

### 1. (Stale-table finding) Best method at the time: h_question | pos | aggregated (78%)
Held under the old code/connectivity conditions. Not yet re-verified under current code — treat as a lead, not a confirmed number.

### 2. retrieve_then_classify beats every re-verified p_question config, and needs no question generation at all
70% vs. p_question's best of 58% — a 12-point gap, with better label balance too (no severe collapse toward one label, unlike p_question's heavy Neutral over-prediction). Root cause, confirmed via direct example inspection: its Locator runs on the **hypothesis itself**, so it reliably finds the one decisive premise sentence a comparison/combination hypothesis depends on. p_question's Stage 1 is **premise-blind** — it never sees the hypothesis at generation time — so it structurally cannot target a hypothesis-specific comparison unless the premise happens to invite one on its own. Confirmed concretely on a real sample (Thomas Young / "surpassed his other skills"): p_question generated 30 real questions, none of which tested the comparison; retrieve_then_classify found the exact sentence directly because it could see what it was looking for.

`retrieve_then_classify` also skips the Answer Extractor entirely — the classifier reads the premise's original wording verbatim instead of a model-generated paraphrase, avoiding a second source of information loss (the Extractor has a separate, confirmed reliability problem with multi-fact arithmetic — see `docs/design/specs/`).

### 3. Removing p_question's Stage 1b relevance filter measurably helped `aggregated` mode
54-58% now vs. 24-46% in the stale table — a large jump. The filter previously capped Stage 1c to only the top-3 questions by ROUGE-L/embedding relevance to H; removing it lets the classifier's own "one decisive piece of evidence is enough" rule (in `prompts/shared_classifier.py`) actually find the needle across the full question pool instead of gambling on a pre-filtered subset.

### 4. Few-shot prompting gives a small, consistent boost to p_question's aggregated mode (+4 points), but doesn't help voting
54%→58% (aggregated) but 30%→24%/22% (voting, across both filter attempts) — few-shot doesn't fix voting's underlying problem, and may make it marginally worse.

### 5. p_question's voting mode is fundamentally broken, and question-filtering doesn't fix it
Three separate approaches tried, none beat the unfiltered 30%/24% baseline: no filter, a relation-type-priority filter (keep all `relation`-tagged questions + top facts), and a pure ROUGE-L relevance filter (drop type entirely, rank all questions by relevance). Root cause, confirmed on a real sample (`id_130`, Bt-resistant insects): the one genuinely decisive question got the correct vote, but was outvoted 9-to-1 by other questions that were each individually, correctly Neutral in isolation (real premise content, just not relevant to *this* hypothesis) — because voting classifies every question with zero visibility into the others. This looks like a limitation of per-question isolated classification itself, not of which questions get selected. An untried alternative: change the vote-*counting* rule (treat Neutral as an abstain rather than a competing vote) instead of filtering the question set — not yet implemented.

### 6. Dominant failure mode across all methods: Neutral over-prediction
Still holds under current code — p_question's aggregated-mode confusion matrix shows heavy Neutral over-prediction on true Entailment/Contradiction cases, consistent with the stale-table finding.

### 7. At least one dataset label looks wrong
`id_109`: gold label is Entailment, but the premise ("Traffic jams... have become a regular feature during monsoon") contains no textual basis whatsoever for the hypothesis (about road-construction material and potholes) — confirmed directly against the raw ConTRoL test.jsonl, not a data-loading bug. Some fraction of "wrong" predictions across every method may be hitting this kind of label noise, an unrecoverable ceiling independent of pipeline quality.

---

## Conclusion (updated 2026-07-22)

The original conclusion — that h_question's question-decomposition approach substantially outperforms direct prompting — likely still holds directionally, but needs re-verification under current code before treating 78% as a real number. The bigger update: **retrieve_then_classify**, the simplest possible evidence-grounded method (no question generation, no Answer Extractor), currently outperforms every re-verified p_question configuration — suggesting that for this dataset/model, decomposing a premise into many small questions may be actively counterproductive, most likely because it fragments the reasoning that the model actually handles better when reading the original located sentences in one piece. Whether this holds up against a re-verified h_question/bridge_question is the open question going into the next round of runs. `p_question`'s voting mode should be considered unresolved/deprioritized rather than fixed by filtering; a real fix, if pursued, likely needs to change the aggregation mechanism itself, not the question selection feeding into it.

**Current status:** a larger production sweep (n=1500, fixed random seed, train split) across `qwen2.5-32b`, `gpt-oss-20b`, and `gemma4-31b` is in progress as of this writing — see `results/production/` once complete for a properly-scaled comparison across all methods and models.
