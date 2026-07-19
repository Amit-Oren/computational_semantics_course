# ConTRoL NLI — Ablation Summary

Model: `qwen2.5-32b` · Dataset: ConTRoL test split · N=50 samples

---

## Pipelines

**zero_shot** — Direct NLI. One call: `(P, H) → label`. No reasoning steps.

**few_shot_cot** — Same but with 3 fixed demonstrations (one per label) in the prompt + chain-of-thought instruction. Forces the model to reason before labeling.

**h_question** — Interrogate H via P.
1. Extract seeds from H (keyphrases)
2. Generate probe questions from those seeds
3. Locate answers in P (Locator: is it answerable? what's the answer?)
4. Classify: given all Q/A pairs → NLI label

**p_question** — Decompose P, test against H.
1. Generate questions *about* P (from decomposed facts or seeded keyphrases)
2. Locate answers in P
3. Filter top-K by relevance to H
4. Classify: given top-K Q/A pairs → NLI label

**bridge_question** — Questions that span both P and H.
1. Generate questions requiring both P and H to answer
2. Locate answers in P
3. Classify from Q/A pairs

---

## Ablation Axes

### Seeder — how seed phrases are extracted from the text
Used by **h_question** (seeds from H) and **p_question|seeded** (seeds from P).

- **`pos`** — NLTK POS tagger extracts noun phrases and named entities. Deterministic, fast, no LLM call. Seeds are surface-level keyphrases (e.g. "corporate taxes", "price levels").
- **`srl`** — LLM extracts PropBank-style semantic roles (agent, theme, time, location…). Produces structured seeds like `"theme: prices  [predicate: fell]"`. Richer than POS but depends on another LLM call.

### Generation — how questions are formed from P
Used by **p_question only**. H is a single sentence so only one approach makes sense there; bridge_question needs no seeder.

- **`decomposition`** — LLM reads the full premise P and breaks it into atomic facts (one clause/claim each), then generates one question per fact. No external seeder needed.
- **`seeded`** — Keyphrases are first extracted from P by the seeder (pos or srl), then passed to the LLM as seeds to generate questions. Two sub-configs: seeded|pos and seeded|srl.

*Why only p_question?* P is a ~452-word narrative with many facts and relations — too complex for one obvious extraction strategy. H is one sentence (only seeding makes sense). Bridge questions are generated freely from the P+H pair (no seeder step at all).*

### Aggregation — how Q/A pairs are combined into a final NLI label
Used by **h_question**, **p_question**, **bridge_question**.

- **`aggregated`** — All Q/A pairs are passed together in a single classifier call. The model sees the full evidence set at once and decides the label. (QAGS / SummaC paradigm.)
- **`sequential_cot`** — Q/A pairs are processed one at a time, left to right. After each pair the model updates a running verdict. Final verdict after the last pair is the label. (Decomposed Prompting paradigm.)
- **`voting`** — Each Q/A pair is classified independently (one call per pair). Final label is the majority vote across all pairs.

zero_shot and few_shot_cot have no ablation axes — single config each.

**Config counts:**
- zero_shot: 1
- few_shot_cot: 1
- bridge_question: 3 (3 aggregations)
- h_question: 6 (2 seeders × 3 aggregations)
- p_question: 9 (decomp×3 + seeded|pos×3 + seeded|srl×3)
- **Total: 20 configs**

---

## Results (N=50, sorted by accuracy)

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

## Key Findings

### 1. Best method: h_question | pos | aggregated (78% accuracy)
- **+14 points** over zero_shot (64%)
- **+8 points** over few_shot_cot (70%)
- Cuts Neutral-fail from 18→9 (50% reduction)
- Contradiction recall: 0.375 (zero_shot) → 0.688 (h_question|pos|agg)

### 2. Seeder: POS beats SRL for h_question
- pos|aggregated: 78% vs srl|aggregated: 74%
- POS is also deterministic (no LLM dependency, reproducible)

### 3. Aggregation: aggregated wins for h_question and bridge_question
- Presenting all Q/A pairs in one call outperforms voting and sequential_cot
- sequential_cot is weakest (degrades as question count grows)

### 4. bridge_question is a strong runner-up (72%)
- Only 6 points behind best h_question config
- No seeder needed — simpler pipeline

### 5. p_question underperforms consistently (24–46%)
- Root cause: questions from P are mostly location-agnostic factual questions
- The top-K relevance filter is too weak — irrelevant Q/A pairs pollute the classifier
- Voting/sequential_cot make it worse (more noise per call)

### 6. Dominant failure mode across all methods: Neutral over-prediction
- zero_shot: 18/50 E/C cases predicted Neutral
- few_shot_cot: 14/50
- h_question|pos|aggregated: 9/50
- Question decomposition is the most effective intervention against this failure mode

---

## Conclusion

Question-based decomposition (h_question) substantially outperforms direct prompting on ConTRoL's multi-step contextual reasoning. The pipeline's core mechanism — locating evidence in P before classifying — directly attacks the model's tendency to predict Neutral when uncertain. POS seeding + aggregated classification is the optimal configuration.
