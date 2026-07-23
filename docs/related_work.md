# Related Work

## How to read this document

Every paper below was re-checked against the **actual running code** — not just
against whether a citation comment exists somewhere. That distinction matters:
several papers are cited in a docstring for a mechanism that the current
production sweep (`run_production_train.py`, n=1,500) never actually executes,
or that turned out to be dead code once traced through. So papers are split
into three tiers:

1. **✅ Confirmed** — cited in the repo, and the mechanism it grounds is part
   of what the current production sweep actually runs.
2. **⚠️ In the code, not active in production** — cited in the repo, but the
   mechanism it grounds is either dead code, unselected by any production
   method, or only ran in earlier small-scale ablations, never at production
   scale. Each row says exactly why.
3. **🚫 Not found in the repo** — supplied for discussion; no docstring,
   comment, or README line cites these. Kept separate so nothing gets
   mistaken for something the code was actually built from.

---

## 1. ✅ Confirmed — actively grounding code in the current production run

| Paper | What it grounds | Where (file) | Production evidence |
|---|---|---|---|
| Liu et al. (2020) — *Natural Language Inference in Context*, arXiv:2011.04864 | The ConTRoL dataset itself | `README.md` | The entire project runs on this dataset. |
| Wang et al. (2020) — QAGS, ACL | The `aggregated` classifier mode — the *only* aggregation mode the production sweep uses | `utils/aggregation.py`, `utils/seeding/pos.py` | `run_production_train.py` hardcodes `aggregation="aggregated"` for every question-based method call (lines 111, 116, 121, 126, 131, 136). |
| Durmus et al. (2020) — FEQA, ACL | Same as above, jointly cited alongside QAGS | `utils/aggregation.py`, `utils/seeding/pos.py` | Same evidence as QAGS. |
| Min et al. (2023) — FActScore, EMNLP | The **fact** half of p_question's `decomposition` mode | `prompts/p_question.py`, `runner/p_question.py` | Production runs `p_question_decomposition` directly. |
| Goyal & Durrett (2020–2021) — DAE | The **relation** half of p_question's `decomposition` mode | `prompts/p_question.py` | Same production method as above. *(This paper has a second citation site that is NOT active — see §2.)* |
| FitzGerald et al. (2018) — QA-SRL | The `srl` seeder | `utils/seeding/srl.py` | Production runs `h_question_srl` and `p_question_seeded_srl`. |
| Pyatkin et al. (2021) — "Asking It All..." | Same seeder, jointly cited | `utils/seeding/srl.py` | Same evidence as QA-SRL. |
| Yang et al. (2018) — HotpotQA, EMNLP | bridge_question's bridging-question requirement | `prompts/bridge_question.py` | Production runs `bridge_question`. |
| Min et al. (2019) — DecompRC, ACL | Same method, jointly cited; also grounds the retry-once-if-no-bridge-question safeguard | `prompts/bridge_question.py` | Same evidence as HotpotQA. |

---

## 2. ⚠️ In the code, but NOT active in the current production run

| Paper | What it nominally grounds | Why it's not actually active right now |
|---|---|---|
| Wei et al. (2022) — Chain-of-Thought Prompting, NeurIPS | `few_shot_cot`, per `README.md` | `prompts/few_shot_cot.py`'s own docstring says: *"Standard 3-shot prompting, not chain-of-thought."* The method has no CoT reasoning step at all — this citation doesn't match what the code does, name notwithstanding. |
| Khot et al. (2023) — Decomposed Prompting | The `sequential_cot` aggregation mode | Confirmed: every production call hardcodes `aggregation="aggregated"` (`run_production_train.py`, lines 111–136). `sequential_cot` only ran in earlier n=50 ablations (26 old result files under `results/`, none under `results/production/`) — never at production scale. |
| Carbonell & Goldstein (1998) — Maximal Marginal Relevance, SIGIR | p_question's `mmr` selection mode | Confirmed dead code: `stage1b_select_questions()` is defined in `runner/p_question.py` but never called anywhere inside `run_sample()`. A comment at `runner/p_question.py:320` confirms Stage 1b was removed outright, replaced by a separate ROUGE-L-only filter (`select_for_voting`) used only in `voting` mode — which is itself not used in production either. |
| Goyal & Durrett (2020–2021) — DAE, *second citation site* | The `svo` seeder | `svo` is registered in `SEEDERS` (`utils/seeding/__init__.py`) but never selected by any production method, and the ablation script's default seeder set is `pos srl` only ("svo needs spaCy," excluded by default in `experiments/ablation_full.py`). DAE's *other* citation — the decomposition-mode relation step — is still active; see §1. |
| Laban et al. (2022) — SummaC, TACL | The Locator, per an earlier version of this document | Re-checked against the actual code: SummaC is cited only as part of the general "QAGS/FEQA/SummaC paradigm" label attached to the **aggregated classifier**, not to the Locator specifically. No code comment anywhere ties SummaC to the Locator module. That mapping was mine, based on how it was described to me — it doesn't hold up against the actual comment. The Locator itself has **no dedicated paper cited for it anywhere in the code**. |
| Kintsch (1988) — Construction-Integration Model, *Psychological Review* | General motivation for question-based decomposition | Cited only in `README.md`'s References list as background theory — not tied to any specific function, mode, or file, so it can't be verified against a concrete code path the way the others above can. |
| Graesser et al. (1994) — Constructing Inferences, *Psychological Review* | Same kind of general motivation | Same caveat as Kintsch (1988) — theoretical framing only, no specific code tie. |

---

## 3. 🚫 Supplied for discussion — not found anywhere in the repo

No docstring, comment, or README line cites any of these. Listed as literature
relevant to the project's design, not as evidence of what the code was built
from.

| Paper | Conceptual relevance |
|---|---|
| IRCoT — Trivedi et al. (2023), "Interleaving Retrieval with Chain-of-Thought Reasoning" | Alternates one reasoning step with one retrieval step. Closest external precedent for `sequential_cot` + per-question Locator calls — though our methods generate all questions upfront rather than letting retrieval steer the next question. |
| Self-Ask — Press et al. (2022) | Asks itself one follow-up question at a time, each based on the prior answer. A *contrast*, not a match: h_question/p_question/bridge_question generate their whole question set upfront in one call, not adaptively. |
| Chen, Lin & Durrett (2019) — "Multi-hop Question Answering via Reasoning Chains" | Graph-structured aggregation of retrieved facts. Conceptually close to how the aggregation stage combines several (question, answer) pairs into one verdict, though nothing here is graph-based. |
| QAFactEval — Fabbri et al., NAACL (2022) | Finding: answerability classification is the most critical component of QA-based metrics. Strong conceptual match for the `answerable`/`NOT_ANSWERABLE` filtering every question-based method applies — the best available justification for why that filtering step matters, even though it's not actually cited in code. |
| Zhou et al. (2022) — Least-to-Most Prompting | Decompose into ordered sub-questions, solve simplest first, feed answers forward — closer to `sequential_cot`'s actual mechanism than Khot et al. (2023), which *is* cited in code (see §2) but isn't itself active in production either way. |
| HotpotQA / WIKIHOP framing for p_question | HotpotQA is cited in-repo, but only for bridge_question (§1) — its use as multi-hop grounding for p_question's "gather evidence, then synthesize" shape, and WIKIHOP (Welbl et al., 2017) alongside it, are not in any p_question-specific code comment. |

---

## Deep dives per method

### bridge_question

| Paper | What bridge_question took from it |
|---|---|
| HotpotQA (Yang et al., 2018) | The bridge/comparison reasoning-type taxonomy — bridge_question hard-requires at least one sub-question to be a bridging or comparison question, mirroring HotpotQA's two dominant multi-hop question categories. |
| DecompRC (Min et al., 2019) | The decompose → answer-each-part → recombine structure — bridge_question generates 2–4 sub-questions from H + P, answers each via the shared Locator/Extractor, then recombines for the final label. |

### p_question

QAGS and FEQA are cited in-repo; QAFactEval and the WIKIHOP framing are not (§3) — this table mixes both tiers deliberately, to show how each paper maps onto a specific part of p_question's design.

| Paper | What it contributed |
|---|---|
| QAGS (Wang, Cho & Lewis, ACL 2020) | Established QA-based factual consistency evaluation. p_question inverts its direction: QAGS generates questions from the summary side, p_question generates them from the premise/source instead. |
| FEQA (Durmus, He & Diab, ACL 2020) | Entity-anchored question generation from the summary side. p_question again inverts this — anchors questions in the premise. |
| QAFactEval (Fabbri et al., NAACL 2022) *(not in repo — §3)* | Answerability classification as the critical component. Conceptual match for p_question's unanswerable-question flagging before classification. |
| HotpotQA (Yang et al., 2018) and WIKIHOP (Welbl et al., 2017) *(this framing not in repo — §3)* | Multi-hop decomposition: gather all supporting facts first, then reason over them as a set — p_question adapts this for NLI classification rather than span extraction. |

### sequential_cot / chaining

**Neither paper here grounds an actively-used production mechanism** — `sequential_cot` never runs in the production sweep (§2). Kept as its own section since it's the clearest place to compare the two chaining papers against each other.

| Paper | Description | Role |
|---|---|---|
| Zhou et al. (2022) — Least-to-Most Prompting *(not in repo — §3)* | Decompose into ordered sub-questions, solve simplest first, feed each answer forward | Closer match to `sequential_cot`'s actual step-by-step mechanism than Khot et al. (2023), which is the one actually quoted in `utils/aggregation.py` |
| Min et al. (2019) — DecompRC | Break a multi-hop claim into a chain of single-hop sub-questions | Already cited in-repo, but for bridge_question (§1) — its "chain" framing applies conceptually here too |

---

## Pipeline part → academic grounding (corrected)

| Pipeline part | Academic grounding | Status |
|---|---|---|
| Locator (finds relevant sentences in P) | **No paper cited in code** — the SummaC mapping in an earlier version of this table didn't hold up on re-check (§2) | — |
| Answer Extractor (reads located sentences → answer) | No explicit citation in the repo | — |
| Classifier, `aggregated` mode | QAGS (Wang et al., 2020) / FEQA (Durmus et al., 2020) / SummaC (Laban et al., 2022, named in-code without full citation) | ✅ active — only mode used in production |
| Classifier, `sequential_cot` mode | Khot et al. (2023) — Decomposed Prompting | ⚠️ not active in production |
| Classifier, `voting` mode | No explicit citation in the repo | ⚠️ not active in production |
| `pos` seeder | QAGS / FEQA | ✅ active |
| `srl` seeder | QA-SRL (FitzGerald et al., 2018); Pyatkin et al. (2021) | ✅ active |
| `svo` seeder | DAE (Goyal & Durrett, 2020–2021) | ⚠️ seeder never selected in production |
| p_question `decomposition` — fact questions | FActScore (Min et al., 2023) | ✅ active |
| p_question `decomposition` — relation questions | DAE (Goyal & Durrett, 2020–2021) | ✅ active |
| `mmr` question selection | Carbonell & Goldstein (1998) | ⚠️ dead code — never called |
| bridge_question | HotpotQA (Yang et al., 2018); DecompRC (Min et al., 2019) | ✅ active |
