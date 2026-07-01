# NLI Methods — Overview

**Task:** Document-level Natural Language Inference on the ConTRoL dataset.  
Each sample has a long premise P (~500 words), a short hypothesis H, and a gold label ∈ {Entailment, Contradiction, Neutral}.  
All methods output a structured JSON result with `metadata` and per-sample fields: `id, premise, hypothesis, label, prediction`.

---

## Baselines

### Zero-Shot

**LLM calls:** 1

The simplest possible approach. The full premise and hypothesis are placed directly into a single prompt and the model is asked to decide the label in one shot — no examples, no decomposition, no intermediate steps.

```
Prompt → [P + H] → LLM → {label, explanation}
```

**Output fields:** `id, label, prediction, explanation`

**When to use as reference:** Sets the lower bound. Any pipeline method should beat this on hard long-document cases.

---

### Few-Shot Chain-of-Thought (few_shot_cot)

**LLM calls:** 1

Same single-call structure as zero-shot, but the prompt is prefixed with **3 fixed demonstrations** — one example of each label (Entailment, Contradiction, Neutral), drawn from the dev split (ids: `id_193`, `id_123`, `id_315`). The demonstrations are manually verified and held constant across all runs to ensure reproducibility.

The model is implicitly nudged to reason step-by-step (chain-of-thought) before outputting a label.

```
Prompt → [3 demonstrations + P + H] → LLM → {label, explanation}
```

**Output fields:** `id, label, prediction, explanation`

**When to use as reference:** Upper bound for single-call approaches. Establishes how much the multi-stage pipelines add beyond in-context learning.

---

## Multi-Stage Pipelines

### q2_pipeline — Query-Based Factual Audit

**LLM calls:** 2

A lightweight two-stage verifier. The hypothesis is interrogated to extract anchors and targeted questions; then a single audit call over the full premise performs structured fact-checking.

#### Stage 1 — Question Generator
Input: H  
Output: a list of **anchor phrases** from H + **2–3 verification questions**, each targeting a single verifiable fact in H.

```
H → LLM → {anchors: [...], questions: ["q1", "q2", "q3"]}
```

#### Stage 2 — Factual Auditor
Input: P + H + the Stage 1 questions  
Output: a structured **audit table** — one row per question — with:
- verbatim premise evidence for each question
- whether the evidence was found (`found: true/false`)
- cross-check flags across questions
- a final NLI label + explanation

```
[P + H + questions] → LLM → {audit_table, matrix_flags, label, explanation}
```

**Decision:** The auditor produces the label directly; no aggregation step needed.

**Output fields:** `id, label, prediction, hypothesis, questions, audit_table, explanation`

---

### p_question — Premise Interrogation Pipeline

**LLM calls:** 3–4 per question (× top-K questions)

Interrogates the premise rather than the hypothesis. The premise is asked what questions it can answer; the most hypothesis-relevant questions are then used to extract evidence and classify.

#### Stage 1a — Question Generation (LLM)
Input: P  
Output: up to **15 factual questions** the premise directly answers, written breadth-first across topics.

```
P → LLM → {questions: ["q1", ..., "q15"]}
```

**NER Coverage Layer (no LLM):** For long premises (>300 words), spaCy NER scans P and adds gap-filling questions for any named entity (person, place, date, quantity) not already covered by the LLM questions. Questions are tagged `[LLM]` or `[NER]` for traceability.

#### Stage 1b — Question Alignment (metric, no LLM)
Input: all questions + H  
Each question is scored against H using **ROUGE-L** (default) or **BLEU**.  
The **top-K** (default K=3) highest-scoring questions are kept.

```
questions × H → ROUGE-L scores → top-3 selected
```

#### Stage 1c — Answer Extraction (2 LLM calls per question)
For each of the top-K questions, a two-step extraction:

**Step 1 — Evidence Gathering:**  
Scan the full P and collect every relevant sentence (verbatim or close paraphrase).

```
[P + question] → LLM → {evidence_sentences: [...], has_evidence: bool}
```

**Step 2 — Answer Synthesis:**  
Summarise the gathered evidence sentences into one clean, final answer.

```
[evidence_sentences + question] → LLM → {answer, unanswerable: bool}
```

#### Stage 2 — NLI Classification (LLM)
Input: all answers + H  
The collected (question, answer) pairs are fed to an NLI classifier to decide the final label.  
Two modes (configurable):
- **`concatenated`** — all Q/A pairs in one call, one label
- **`majority_vote`** — one call per Q/A pair, label by vote

```
[Q/A pairs + H] → LLM → {label}
```

**Output fields:** `id, label, prediction, stage1a_questions, stage1a_sources, stage1b_aligned, stage1c_answers, stage2_label`

---

### h_question — Hypothesis Interrogation Pipeline

**LLM calls:** 1 + 2 per probe question + 1 (judge)

Inverts p_question: instead of asking what the premise covers, it asks what the hypothesis needs the premise to confirm. Probe questions are generated from H, then each is answered from P, and a holistic judge decides the final label.

#### Stage 0 — Keyphrase Extraction (no LLM)
Input: H  
NLTK POS tagging extracts noun phrases, main verbs, and standalone numerals as **keyphrases**.

```
H → POS tagger → {keyphrases: [...], claim_from_H: str}
```

#### Stage 1 — Probe Question Generation (LLM)
Input: H + keyphrases  
Output: **1–2 targeted probe questions** that, when answered from P, will reveal whether H is entailed, contradicted, or neutral. Each question targets exactly one verifiable claim.

```
[H + keyphrases] → LLM → {questions: ["q1", "q2"]}
```

#### Stage 2a — Sentence Locator (LLM, per question)
Input: numbered P (each sentence indexed `[0], [1], ...`) + probe question  
Output: up to 5 **sentence indices** in P most relevant to answering the question (multi-hop allowed).

```
[numbered P + question] → LLM → {indices: [2, 7, 11]}
```

#### Stage 2b — Answer Extractor (LLM, per question)
Input: extracted sentences + probe question  
Output: a **one-sentence answer** using only the located sentences. Returns `NOT_ANSWERABLE` only if the sentences contain no related information at all (allowed to make one small, obvious inference).

```
[sentences + question] → LLM → {answer: str}
```

#### Stage 3 — Holistic Judge (LLM)
Input: H + all (question, answer) pairs as a structured probes block  
The judge sees the full picture at once and decides ONE label, reasoning over all probes together. Scope rules: partial or topic-specific evidence does not entail "overall/same/always" claims.

```
[H + probes block] → LLM → {label}
```

**Output fields:** `id, label, prediction, keyphrases, gen_questions, per_question_details[]`  
Each `per_question_details` entry: `question, located_indices, extracted_sentences, answer_from_P, answerable_flag`

---

### h_multihop — Atomic Sub-Question Chaining

**LLM calls:** 1 + 2 per hop (max 3 hops) + 1 (classifier)

The most structurally complex method. Rather than generating independent parallel questions, it builds a **sequential chain** of atomic sub-questions where each hop's answer becomes context for the next. Designed to handle multi-step reasoning (e.g. "does entity X exist?" → "what did X do?" → "does that match H?").

#### Stage 0 — Keyphrase Extraction (no LLM)
Same as h_question: NLTK POS tagging extracts keyphrases from H.

#### Stage 1 — Decomposition Planner (LLM)
Input: H + keyphrases  
Output: **2–3 ordered atomic sub-questions**, designed so the simplest/most-presupposed fact is checked first.

Ordering rules enforced by the prompt:
- **Existence first:** if H assumes an entity exists (e.g. "his mother"), the first sub-question must verify that entity exists in P before asking what it did.
- Each sub-question uses the exact names and qualifiers from H (never replaced with generic words).
- Maximum 3 sub-questions.

```
[H + keyphrases] → LLM → {sub_questions: ["q1", "q2", "q3"]}
```

#### Stage 2 — Sequential Answering Loop (LLM × 2 per hop, max 3 hops)

For each sub-question **in order**:

**Stage 2a — Sentence Locator (LLM):**  
Same locator as h_question — returns up to 5 sentence indices from numbered P.

**Stage 2b — Per-hop Answer Extractor (LLM):**  
Answers the sub-question using extracted sentences **plus the running context** (all prior sub-questions and their answers). Allowed one small inference. Returns `NOT_ANSWERABLE` only if sentences have no related information.

```
[prior Q/A context + sub-question + sentences] → LLM → {answer, answerable: bool}
```

**Halting rule:** If a hop returns `NOT_ANSWERABLE`, the chain stops asking further sub-questions — but does **not** automatically set the label to Neutral. All hops that were answered are still passed to the classifier.

Running context after each answered hop:
```
Q1: <sub_question_1>
A1: <answer_1>

Q2: <sub_question_2>
A2: <answer_2>
```

#### Stage 3 — Chain Classifier (LLM)
Input: H + the **answered hops** as a structured chain block (partial chains are valid)  
Decides ONE label. A single answered hop that proves a conflict → Contradiction even if later hops were unanswerable.

```
[H + chain block (answered hops only)] → LLM → {label}
```

**Fallback:** If NO hop was answered at all → Neutral (classifier not called).

**Output fields:** `id, label, prediction, keyphrases, gen_questions, per_question_details[], chain[]`  
`per_question_details` mirrors h_question shape for backward compatibility.  
`chain[]` adds per-hop fields: `hop_index, sub_question, context_at_hop, located_indices, extracted_sentences, answer_from_P, answerable_flag, halted`

---

## Summary Table

| Method | LLM Calls | Interrogates | Key Idea |
|---|---|---|---|
| zero_shot | 1 | — | Direct classification, no decomposition |
| few_shot_cot | 1 | — | 3 fixed demonstrations guide reasoning |
| q2_pipeline | 2 | H | Structured audit table per question |
| p_question | 3–4 × K | P | Premise-generated questions filtered by alignment to H |
| h_question | 1 + 2K + 1 | H | Parallel probe questions answered from P, holistic judge |
| h_multihop | 1 + 2K + 1 | H | Sequential chain: each hop's answer feeds the next |

*K = number of questions/hops (p_question default K=3, h_question 1–2, h_multihop max 3)*

---

## Shared Infrastructure

- **Model backend:** OpenAI-compatible API (`ChatOpenAI`) — lab server via Tailscale, HuggingFace, or Groq
- **Structured output:** All LLM calls use `with_structured_output` (Pydantic schemas) — no manual JSON parsing
- **Sentence indexing:** `utils/premise_indexer.py` — NLTK sentence tokenizer, `[i] sentence` format
- **Keyphrase extraction:** `utils/pos_keyphrase.py` — NLTK RegexpParser for NP/VB/CD chunks
- **Retry logic:** All LLM calls wrapped in `_call_with_retry` (5 attempts, 30s wait on 429/503)
- **Results:** Saved to `results/{experiment}_{model}_{timestamp}.json`
- **Logs:** Per-run log to `logs/{experiment}_{model}_{timestamp}.log`
