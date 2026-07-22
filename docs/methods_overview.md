# NLI Methods — Overview

**Task:** Document-level Natural Language Inference on the ConTRoL dataset.
Each sample has a long premise P (~500 words), a short hypothesis H, and a gold label ∈ {Entailment, Contradiction, Neutral}.

**Note on scope:** two earlier methods, `q2_pipeline` and `h_multihop`, were archived before this document's last rewrite and are not part of the current core method set — see `experiments/archived/` if they're needed for reference. This document covers the 6 methods currently in `runner/`.

---

## Baselines

### Zero-Shot (`zero_shot`)

**LLM calls:** 1

The full premise and hypothesis go directly into a single prompt; the model decides the label in one shot — no examples, no decomposition, no intermediate steps.

```
[P + H] → LLM → {label, explanation}
```

**When to use as reference:** Sets the lower bound. Any pipeline method should beat this.

---

### Few-Shot Chain-of-Thought (`few_shot_cot`)

**LLM calls:** 1

Same single-call structure as zero-shot, prefixed with **3 fixed demonstrations** — one per label — drawn from the dev split (`id_193`, `id_123`, `id_315`), held constant across all runs.

```
[3 demonstrations + P + H] → LLM → {label, explanation}
```

**When to use as reference:** Upper bound for single-call approaches — shows how much the multi-stage pipelines add beyond in-context learning.

---

## Control Baseline

### Retrieve-Then-Classify (`retrieve_then_classify`)

**LLM calls:** 2 (no question generation, no Answer Extractor)

Isolates the Locator's contribution from question generation. The Locator runs directly on the **hypothesis itself** as its query (not a generated question), pulling up to 5 relevant premise sentences; those raw sentences (not a paraphrased "answer") go straight to the classifier.

```
Stage 1: [numbered P + H as query] → Locator (LLM) → {indices}
Stage 2: [raw located sentences + H] → Classifier (LLM) → {label}
```

Purpose: `zero_shot → retrieve_then_classify` gap = value of locating alone. `retrieve_then_classify → question-based methods` gap = value actually added by generating questions. In practice this has been the strongest-performing method tested so far (70% on an n=50 diagnostic slice, ahead of every `p_question` configuration) — because it's the only method whose Locator sees the hypothesis directly, and the only one that never risks the Answer Extractor rephrasing/losing precision from the original sentence.

**No aggregation axis, no few-shot axis** — it's a fixed, minimal 2-stage method by design.

---

## Multi-Stage Pipelines

All three question-based pipelines below (`h_question`, `p_question`, `bridge_question`) share the same evidence-finding and classification infrastructure — only Stage 1 (question generation) differs. See **Shared Infrastructure** at the end.

### h_question — Hypothesis Interrogation (`runner/h_question.py`)

Generates probe questions from the hypothesis (premise-blind at generation time), then locates/answers/classifies via the shared pipeline.

**Stage 0 — Keyphrase extraction (no LLM):** NLTK POS tagging (`pos` seeder) or LLM-based semantic-role extraction (`srl` seeder) pulls keyphrases from H.

**Stage 1 — Probe question generation (LLM):** given H + keyphrases, generates **2-4** targeted probe questions. Includes a comparison/scope rule (hypotheses with "more/less/than", "always/never" etc. must get at least one question testing the comparison directly, not just per-entity yes/no questions) and an information-integration rule (hypotheses requiring combining two facts get one combined question, not two isolated ones — this rule is present under the label `INFORMATION INTEGRATION` inline in the prompt).

```
[H + keyphrases] → LLM → {questions: [...]}
```

**Stages 2-4 — shared** (see Shared Infrastructure): locate + extract per question, then classify.

**Axes:** `seeder_name` (`pos`/`srl`), `aggregation` (`aggregated`/`sequential_cot`/`voting`), `few_shot` (on/off — 3 worked examples appended to the Stage 1 prompt).

---

### p_question — Premise Interrogation (`runner/p_question.py`)

Generates questions from the premise (**premise-blind** — never sees H at generation time), then locates/answers/classifies. This premise-blindness is a known, confirmed limitation: it structurally cannot target a hypothesis-specific comparison unless the premise happens to invite one on its own (see `bridge_question` below, designed specifically to fix this).

**Stage 1a — Question generation (LLM), 3 generation modes:**
- `decomposition` (default) — decomposes P into atomic **facts** and **relations** (comparisons, causal links, evaluative claims, temporal ordering), tagging each question `"fact"` or `"relation"`.
- `seeded` — a seeder (`pos` or `srl`) extracts keyphrases/anchors from P first, then one question is generated per anchor.
- `freeform` (legacy baseline) — up to 15 breadth-first questions, no fact/relation distinction.

**NER coverage layer (no LLM):** for premises over 300 words, spaCy NER scans P and adds gap-filling questions for any named entity not already covered.

**Capping:** the combined question pool is capped at `P_QUESTION_MAX_QUESTIONS` (currently 15), keeping relation-type questions first (scarce, higher-value) and filling the remainder with facts.

**Stage 1b (selection) is currently disabled** for `aggregated`/`sequential_cot` — all generated questions go directly to Stage 1c unfiltered (removing the old top-K relevance filter measurably improved `aggregated` mode's accuracy: 54-58% vs. 24-46% under the old filtered code, on the same n=50 slice).

**Voting mode has its own separate filter** (`utils/question_selectors.py::select_for_voting`), because `voting`'s per-question isolated classification collapses toward Neutral when diluted by many low-signal questions (confirmed: 30% unfiltered, and two different filtering strategies — relation-type priority, then pure ROUGE-L relevance ranking — both failed to improve on that baseline, at 30% and 24% respectively, on an n=50 diagnostic). The filter still exists in code (`voting_cap`, default 10) but should be treated as **unresolved** — no filtering approach tried so far has fixed voting mode for `p_question`; the underlying problem looks like the per-question isolated-classification mechanism itself, not question selection.

**Stages 1c-2 — shared** (see Shared Infrastructure).

**Axes:** `generation` (`decomposition`/`seeded`/`freeform`), `seeder_name` (for seeded mode), `aggregation`, `few_shot`.

---

### bridge_question — Both-Texts Bridging (`runner/bridge_question.py`)

Generates questions from **both P and H together** — the only question-based method whose Stage 1 sees both texts, specifically to fix the premise-blind/hypothesis-blind gap the other two methods have.

**Stage 1 (LLM):** given P + H, generates **2-4** sub-questions; at least one must be a bridging/comparison question testing the hypothesis's **core claim** directly (identified explicitly as a first step in the prompt). A retry fires once if no bridging question comes out on the first attempt. Includes the same information-integration rule as `h_question` (labeled `INFORMATION INTEGRATION` in this file), with a worked ✗/✓ example since it can see the premise too.

```
[P + H] → LLM → {questions: [...], bridge_indices: [...]}
```

**Stages 2-4 — shared** (see Shared Infrastructure).

**Axes:** `aggregation`, `few_shot`. No seeder axis (doesn't need one — it sees both texts directly).

---

## Shared Infrastructure

Used identically by `h_question`, `p_question`, and `bridge_question` — only question generation (Stage 1) differs between them.

- **Locator** (`utils/locator_extractor.py::locate_and_answer`) — numbered premise + question → up to 5 sentence indices (multi-hop allowed), plus a one-sentence `reasoning` field.
- **Answer Extractor** — extracted sentences + question → a concise answer, using only those sentences (at most one small, obvious inference allowed). Returns `NOT_ANSWERABLE` only when the sentences contain no related information at all.
- **Classifier** (`prompts/shared_classifier.py::classify_evidence`) — evidence + hypothesis → one NLI label. Priority order: Contradiction → Entailment → Neutral; explicitly instructed not to let weak/tangential evidence "average away" one independently decisive piece.
- **Aggregation modes** (`utils/aggregation.py`), used by all three question-based methods:
  - `aggregated` — all (question, answer) pairs in a single classifier call.
  - `sequential_cot` — pairs processed one at a time, running verdict updated after each.
  - `voting` — one classifier call per pair, majority label wins (ties broken by the same Contradiction > Entailment > Neutral priority). **Known to underperform significantly** for `p_question` (see above) — not yet root-caused beyond "per-question isolation dilutes the few decisive votes among many correctly-uninformative ones."
- **Model backend:** OpenAI-compatible API (`ChatOpenAI`), routed via `config.config.get_llm`/`get_structured_llm` — lab server (self-hosted open-weight models via an Ollama-backed gateway), HuggingFace, or Groq depending on the model.
- **Structured output:** all "open_source" (lab-hosted) model calls go through a custom `_StructuredOutput` wrapper (`config/config.py`) — strips markdown fences and repairs a recurring Ollama formatting slip (unquoted enum-like values, e.g. `"type": fact` instead of `"type": "fact"`) before Pydantic validation. Other providers use LangChain's `with_structured_output` directly.
- **Retry logic:** `utils/retry.py::call_with_retry` — up to 5 attempts, 30s wait, only on errors matching `capacity/rate limit/429/503/overloaded/timeout` substrings (a bare connection reset is *not* retried — it's treated as a same-call failure and falls through to that stage's own fallback).
- **Zero-shot fallback:** every question-based method falls back to a direct zero-shot P+H call if Stage 1 produces no questions, or if every generated question turns out unanswerable — check a result's `warnings` field for "Stage 1a/1 returned no questions" or "zero-shot fallback" before trusting its prediction as reflecting that method's actual mechanism.
- **Sentence indexing:** `utils/premise_indexer.py` — NLTK sentence tokenizer, `[i] sentence` format.
- **Keyphrase extraction:** `utils/pos_keyphrase.py` (POS/NLTK) and `utils/seeding/` (POS/SRL seeders used by `h_question` and `p_question`'s seeded mode).
- **Results:** saved to `results/{experiment}_{model}_{timestamp}.json` (or `ablation_{tag}_{model}_{timestamp}.json` for ablation-script runs).
- **Logs:** per-run log to `logs/{experiment}_{model}_{timestamp}.log`.

---

## Summary Table

| Method | LLM calls | Sees at generation | Aggregation axis | Few-shot axis | Key idea |
|---|---|---|---|---|---|
| `zero_shot` | 1 | — | — | — | Direct classification, no decomposition |
| `few_shot_cot` | 1 | — | — | — | 3 fixed demonstrations guide reasoning |
| `retrieve_then_classify` | 2 | H (as Locator query) | — | — | Locator-only control; no question generation |
| `h_question` | 1 + 2 per question + 1 | H only | ✓ | ✓ | Hypothesis-generated probes, premise-blind at generation |
| `p_question` | 1 + 2 per question + 1 | P only | ✓ | ✓ | Premise-generated questions; known premise-blind gap |
| `bridge_question` | 1 + 2 per question + 1 | Both P and H | ✓ | ✓ | Only method whose Stage 1 sees both texts together |
