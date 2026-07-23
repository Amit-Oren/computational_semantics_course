# Related Work

Every entry below is grounded in a real citation already present in this repo's
code comments/docstrings, or in `README.md`'s References section — see the
file path noted per row. The exception is explicitly marked.

## All papers referenced in the project

| Paper | Where it's cited in the repo | What we took from it |
|---|---|---|
| Liu et al. (2020) — *Natural Language Inference in Context*, arXiv:2011.04864 | `README.md` | The source paper — this project replicates/extends its task: NLI over long, expert-designed premises using the ConTRoL dataset it introduced. |
| Wei et al. (2022) — Chain-of-Thought Prompting, NeurIPS | `README.md` | Showing worked reasoning examples before asking the model to answer improves multi-step reasoning — basis for the `few_shot_cot` baseline. |
| Kintsch (1988) — Construction-Integration Model, *Psychological Review* | `README.md` | Cognitive theory of building a mental model of text by integrating separate propositions — motivation for decomposing a premise instead of reading it monolithically. |
| Graesser et al. (1994) — Constructing Inferences During Narrative Text Comprehension, *Psychological Review* | `README.md` | Theory of the inferences human readers generate while reading — motivates the general "ask probing questions" strategy behind h_question/p_question/bridge_question. |
| Wang et al. (2020) — QAGS, ACL | `utils/seeding/pos.py`, `utils/aggregation.py` | Verify a claim by generating questions and checking answers against source text — grounds the question→answer→classify pipeline shape and the `aggregated` classification mode. |
| Durmus et al. (2020) — FEQA, ACL | `utils/seeding/pos.py`, `utils/aggregation.py` | Same QA-based faithfulness-checking family as QAGS — joint grounding for answer-first, keyphrase-anchored question generation. |
| Laban et al. (2022) — SummaC, TACL | `utils/aggregation.py`, `experiments/ablation_full.py` (named only, no author/year in-code — full citation supplied outside the repo) | Sentence-level NLI/evidence localization — grounds the Locator's role of finding relevant sentences before any answer is extracted. |
| Min et al. (2023) — FActScore, EMNLP | `prompts/p_question.py`, `runner/p_question.py` | Breaking text into the smallest independently-checkable factual claims ("atomic facts") — basis for the **fact** half of p_question's `decomposition` mode. |
| Goyal & Durrett (2020–2021) — DAE (Dependency Arc Entailment) | `prompts/p_question.py`, `utils/seeding/svo.py` | Relation-level entailment checking (verifying connections between facts, not just facts in isolation) — basis for the **relation** half of `decomposition` mode. |
| Khot et al. (2023) — Decomposed Prompting | `utils/aggregation.py` | Feeding evidence one step at a time with a running, updatable answer — design of the `sequential_cot` aggregation mode. |
| Carbonell & Goldstein (1998) — Maximal Marginal Relevance, SIGIR | `utils/question_selectors.py` | Greedily picking a set that balances relevance against redundancy with what's already picked — basis for the `mmr` question-selection mode. |
| FitzGerald et al. (2018) — QA-SRL | `utils/seeding/srl.py` | Representing a sentence's semantic roles (who did what, to whom) in a QA-friendly format — foundational idea behind the `srl` seeder. |
| Pyatkin et al. (2021) — "Asking It All: Generating Contextualized Questions for any Semantic Role" | `utils/seeding/srl.py` | Extends QA-SRL toward one full contextualized question per semantic role — the SRL seeder's output shape (one seed per role-argument pair). |
| Yang et al. (2018) — HotpotQA, EMNLP | `prompts/bridge_question.py` | Multi-hop QA requiring info from two separate places — grounds bridge_question's core idea of a question needing both texts at once. |
| Min et al. (2019) — DecompRC, ACL *(different Min et al. than the 2023 FActScore paper above)* | `prompts/bridge_question.py` | Decomposing multi-hop questions into sub-questions, and the observation that decomposition models tend to shortcut a multi-hop question into a single-hop guess — bridge_question's retry-once safeguard. |

---

## bridge_question deep dive

| Paper | What bridge_question took from it |
|---|---|
| HotpotQA (Yang et al., 2018) | The bridge/comparison reasoning-type taxonomy — bridge_question hard-requires at least one sub-question to be a bridging or comparison question, mirroring HotpotQA's two dominant multi-hop question categories. |
| DecompRC (Min et al., 2019) | The decompose → answer-each-part → recombine structure — bridge_question generates 2–4 sub-questions from H + P, answers each via the shared Locator/Extractor, then recombines for the final label. |

---

## p_question deep dive

Note: QAGS and FEQA are cited in-repo (`utils/seeding/pos.py`, `utils/aggregation.py`), and HotpotQA is cited in-repo for bridge_question (`prompts/bridge_question.py`) — but its use here, and QAFactEval's and WIKIHOP's, describe p_question's design conceptually rather than quoting a p_question-specific comment in the code.

| Paper | What it contributed |
|---|---|
| QAGS (Wang, Cho & Lewis, ACL 2020) | Established the QA-based factual consistency evaluation paradigm — generate questions, check if answers match. p_question inverts its direction: QAGS generates questions from the summary (the hypothesis-equivalent side), p_question generates them from the premise/source instead. |
| FEQA (Durmus, He & Diab, ACL 2020) | Entity-anchored question generation from the summary side, for faithfulness scoring. p_question again inverts this — anchors questions in the premise rather than the summary/hypothesis. |
| QAFactEval (Fabbri et al., NAACL 2022) | Identified answerability classification as the most critical component of QA-based metrics. This is where p_question's [UNANSWERABLE] flagging in Stage 1c comes from directly — if a generated question has no supporting evidence in the premise, it's flagged and excluded rather than forced into an answer. |
| HotpotQA (Yang et al., 2018) and WIKIHOP (Welbl et al., 2017) | Multi-hop QA decomposition strategy: gather all supporting facts first, then reason over them together as a set. p_question adapts this for NLI classification (Stage 1c's "gather evidence, then synthesize" two-step) rather than span extraction. |

---

## sequential_cot / chaining deep dive

Zhou et al. (2022) is not cited anywhere in the repo — added here as an external grounding, marked Core since it's a closer match to `sequential_cot`'s actual mechanism than Khot et al. (2023), which is the one currently quoted in `utils/aggregation.py`.

| Paper | Description | Role |
|---|---|---|
| Zhou et al. (2022) — Least-to-Most Prompting | Decompose a problem into ordered sub-questions, solve simplest first, feed each answer forward | ✅ Core — this is the backbone of the chaining |
| Min et al. (2019) — DecompRC ("Multi-hop RC through Question Decomposition") | Break a multi-hop claim into a chain of single-hop sub-questions | Supporting — same decompose-then-chain shape, already cited in-repo for bridge_question |

---

## Pipeline part → academic grounding

| Pipeline part | Academic grounding |
|---|---|
| Locator (finds relevant sentences in P) | SummaC (Laban et al., TACL 2022) — sentence-level NLI/evidence localization |
| Answer Extractor (reads located sentences → answer) | No explicit citation in the repo — built as supporting infrastructure around the Locator/Classifier, not modeled on a specific paper |
| Classifier, `aggregated` mode (all Q/A pairs → one call) | QAGS (Wang et al., 2020) / FEQA (Durmus et al., 2020) / SummaC (Laban et al., 2022) — the shared QA-based faithfulness-checking paradigm |
| Classifier, `sequential_cot` mode (one Q/A at a time, running verdict) | Khot et al. (2023) — Decomposed Prompting |
| Classifier, `voting` mode (one call per Q/A pair, majority wins) | No explicit citation in the repo |
| `pos` seeder (NLTK keyphrases, no LLM) | QAGS / FEQA — answer-first, keyphrase-anchored question generation |
| `srl` seeder (LLM-based semantic roles) | QA-SRL (FitzGerald et al., 2018); Pyatkin et al. (2021) |
| `svo` seeder (spaCy dependency triples) | DAE (Goyal & Durrett, 2020–2021) |
| p_question `decomposition` mode — fact questions | FActScore (Min et al., 2023) |
| p_question `decomposition` mode — relation questions | DAE (Goyal & Durrett, 2020–2021) |
| `mmr` question selection | Carbonell & Goldstein (1998) |
| bridge_question (bridging/comparison questions, decompose→answer→recombine) | HotpotQA (Yang et al., 2018); DecompRC (Min et al., 2019) |

---

## Additional related work (not cited in the repo)

These do **not** appear anywhere in the codebase — no docstring, comment, or
README line references them. Listed here as literature relevant to the
project's design, not as evidence of what the code was actually built from.
(HotpotQA and DecompRC, which you also listed, are already covered above —
they *are* cited in `prompts/bridge_question.py`.)

| Paper | Conceptual relevance to this project |
|---|---|
| IRCoT — Trivedi et al. (2023), "Interleaving Retrieval with Chain-of-Thought Reasoning" | Alternates one reasoning step with one retrieval step, repeatedly. Closest external precedent for `sequential_cot` aggregation mode + the shared Locator being called fresh per question — though our methods generate all questions upfront rather than letting retrieval steer the next question, which IRCoT does. |
| Self-Ask — Press et al. (2022) | The model asks itself one follow-up question at a time, each based on the answer to the previous one, before giving a final answer. A useful *contrast*, not a match: our h_question/p_question/bridge_question generate their whole question set upfront in one Stage 1 call, rather than adaptively one hop at a time the way Self-Ask does. |
| Chen, Lin & Durrett (2019) — "Multi-hop Question Answering via Reasoning Chains" | Graph-structured aggregation of multiple retrieved facts into one reasoning chain before answering. Conceptually close to what the aggregation stage does structurally (combining several independently-retrieved (question, answer) pairs into one final verdict), though this project's aggregation modes (aggregated/sequential_cot/voting) aren't graph-based. |
| QAFactEval — Fabbri et al., NAACL (2022) | Finding: answerability classification is the single most critical component of QA-based faithfulness metrics — not the question generation or the answer comparison. Directly relevant to this project's `answerable` field (the Answer Extractor's `{answer, answerable}` output) and the `NOT_ANSWERABLE` filtering step every question-based method applies before classification — this paper is arguably the strongest available justification for why that filtering step matters, closing the gap flagged as "no explicit citation" for the Answer Extractor row above. |
