"""
ablation_p_question_selectors.py — Run p_question across the Stage 1b
ablation grid: {rouge_l, embedding, llm_relevance} relevance scorers x
{topk, mmr} selection modes, on the same fixed held-out test-split dev set.
Reports accuracy + per-label precision/recall/F1 for each config —
Contradiction recall is the key metric, since that's where the ROUGE-L/topk
baseline was failing.

For every sample where gold is Entailment/Contradiction but the prediction is
Neutral, also diagnoses WHY, by re-running locate_and_answer on every
Stage-1a question that Stage 1b did NOT select, then re-classifying with the
full question set:
  - "no_answerable_question_generated": nothing in Stage 1a was answerable at
    all — a Stage 1a generation gap, upstream of selection.
  - "dropped_by_selector": using every generated question flips the
    prediction to gold, and at least one of the flipping questions was
    dropped in Stage 1b — the selector/selection is the bottleneck. Compare
    this rate between topk and mmr for the same scorer: MMR should reduce it
    if the coverage hypothesis is correct.
  - "insufficient_even_with_all_questions": even the full question set
    doesn't recover the gold label — Stage 1a never generated the decisive
    question (or Stage 1c/2 misjudged it), not a Stage 1b problem.
  - "kept_but_misclassified": everything needed was already selected and
    answered; Stage 1b is not the bottleneck (classifier issue instead).

Usage:
    python ablation_p_question_selectors.py --model <model> [--n 20] [--top_k 6]
        [--selectors rouge_l embedding] [--selections topk mmr] [--mmr_lambda 0.7]
"""

import argparse
import json
import os
from datetime import datetime

from config.config import (
    DEFAULT_PARAMS, RESULTS_DIR, P_QUESTION_MMR_LAMBDA, P_QUESTION_MAX_QUESTIONS,
    setup_logger,
)
from evaluate_shared_pipeline import pick_dev_set, precision_recall_f1
from prompts.shared_classifier import classify_evidence
from runner import p_question
from utils.locator_extractor import locate_and_answer
from utils.premise_indexer import number_sentences
from utils.question_selectors import SELECTORS, SELECTION_MODES

FAILURE_GOLD_LABELS = ("Entailment", "Contradiction")


def diagnose_sample(sample: dict, result: dict, model: str, params: dict) -> str:
    """Diagnose why a gold Entailment/Contradiction sample predicted Neutral."""
    premise    = sample["premise"]
    hypothesis = sample["hypothesis"]
    gold       = result["gold_label"]

    selected_questions = {item["question"] for item in result["selected"]}
    kept_qa = list(result["qa_pairs"])  # already-answerable, already-selected pairs

    # all_questions is a list of {"q", "type"} dicts (atomic-fact + relation
    # decomposition) — extract the question text.
    all_q_texts = [d["q"] if isinstance(d, dict) else d for d in result["all_questions"]]
    unselected = [q for q in all_q_texts if q not in selected_questions]

    if not unselected:
        # Nothing was dropped — Stage 1b selected everything Stage 1a generated.
        return "kept_but_misclassified"

    indexed_sentences, numbered_premise = number_sentences(premise)
    extra_qa = []
    for q in unselected:
        r = locate_and_answer(
            model, params, q,
            indexed_sentences=indexed_sentences, numbered_premise=numbered_premise,
        )
        if r["answerable"]:
            extra_qa.append(r)

    all_qa = kept_qa + extra_qa
    if not all_qa:
        return "no_answerable_question_generated"

    full_prediction = classify_evidence(model, params, all_qa, hypothesis)
    if full_prediction == gold:
        dropped_decisive = [qa for qa in extra_qa]  # any of these joined kept_qa to flip it
        if dropped_decisive:
            return "dropped_by_selector"
        return "kept_but_misclassified"

    return "insufficient_even_with_all_questions"


def run_config(
    selector: str, selection: str, dev_set: list[dict], model: str, params: dict,
    top_k: int, mmr_lambda: float, max_questions: int, diagnose: bool = True,
) -> list[dict]:
    tag = f"selector={selector} selection={selection}" + (f" lambda={mmr_lambda}" if selection == "mmr" else "")
    print(f"\n{'=' * 60}\n  Running p_question — {tag}\n{'=' * 60}")
    results = p_question.run(
        dev_set, model=model, params=params,
        selector=selector, selection=selection, top_k=top_k, mmr_lambda=mmr_lambda,
        max_questions=max_questions,
    )

    if not diagnose:
        return results

    by_id = {s["id"]: s for s in dev_set}
    for r in results:
        if r["gold_label"] in FAILURE_GOLD_LABELS and r["prediction"] == "Neutral":
            diagnosis = diagnose_sample(by_id[r["id"]], r, model, params)
            r["diagnosis"] = diagnosis
    return results


def report(tag: str, results: list[dict]) -> dict:
    n = len(results)
    correct = sum(r["gold_label"] == r["prediction"] for r in results)
    accuracy = correct / n if n else 0.0

    print(f"\n{'=' * 60}\n  {tag}  ({n} samples)\n{'=' * 60}")
    print(f"  Accuracy: {correct}/{n} = {accuracy:.2%}")

    stats = precision_recall_f1(results)
    print(f"\n  {'Label':<16} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 54)
    for lbl in ["Entailment", "Contradiction", "Neutral"]:
        s = stats[lbl]
        marker = "  <-- key metric" if lbl == "Contradiction" else ""
        print(f"  {lbl:<16} {s['support']:>8} {s['precision']:>9.2%} {s['recall']:>7.2%} {s['f1']:>7.2%}{marker}")

    diagnosed = [r for r in results if "diagnosis" in r]
    dropped_count = sum(1 for r in diagnosed if r["diagnosis"] == "dropped_by_selector")
    if diagnosed:
        print(f"\n  Gold Entailment/Contradiction predicted Neutral: {len(diagnosed)}  "
              f"(dropped_by_selector: {dropped_count})")
        for r in diagnosed:
            print(f"    id={r['id']}  gold={r['gold_label']}  diagnosis={r['diagnosis']}")
    print()

    return {
        "accuracy": accuracy,
        "contradiction_recall": stats["Contradiction"]["recall"],
        "dropped_by_selector_count": dropped_count,
        "failure_count": len(diagnosed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--mmr_lambda", type=float, default=P_QUESTION_MMR_LAMBDA)
    parser.add_argument("--max_questions", type=int, default=P_QUESTION_MAX_QUESTIONS)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--selectors", nargs="+", choices=SELECTORS, default=list(SELECTORS),
                         help="Which relevance scorers to include in the grid (default: all three)")
    parser.add_argument("--selections", nargs="+", choices=SELECTION_MODES, default=list(SELECTION_MODES),
                         help="Which selection modes to include in the grid (default: both)")
    parser.add_argument("--repeats", type=int, default=1,
                         help="Run each config this many times and report the accuracy spread "
                              "(diagnosis only runs on the first repeat, to bound cost)")
    args = parser.parse_args()

    setup_logger("p_question_selector_ablation", args.model)

    params = dict(DEFAULT_PARAMS)
    if args.max_tokens is not None:
        params["max_tokens"] = args.max_tokens

    dev_set = pick_dev_set(args.n)
    dev_ids = [s["id"] for s in dev_set]
    print(f"Pinned dev set ({len(dev_ids)} ids, from held-out test split): {dev_ids}")
    print(f"Grid: {args.selectors} x {args.selections}, top_k={args.top_k}, "
          f"mmr_lambda={args.mmr_lambda}, max_questions={args.max_questions}, repeats={args.repeats}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "-")

    all_results = {}
    summaries = {}
    spreads: dict[str, list[float]] = {}
    for selector in args.selectors:
        for selection in args.selections:
            base_tag = f"{selector}_{selection}"
            accuracies = []
            for rep in range(args.repeats):
                tag = base_tag if args.repeats == 1 else f"{base_tag}_rep{rep + 1}"
                results = run_config(
                    selector, selection, dev_set, args.model, params,
                    top_k=args.top_k, mmr_lambda=args.mmr_lambda,
                    max_questions=args.max_questions, diagnose=(rep == 0),
                )
                all_results[tag] = results
                accuracies.append(sum(r["gold_label"] == r["prediction"] for r in results) / len(results))

                out_path = os.path.join(
                    RESULTS_DIR, f"p_question_{tag}_{safe_model}_devset_{timestamp}.json"
                )
                with open(out_path, "w") as f:
                    json.dump({
                        "metadata": {"experiment": "p_question", "selector": selector, "selection": selection,
                                     "top_k": args.top_k, "mmr_lambda": args.mmr_lambda,
                                     "max_questions": args.max_questions, "model": args.model,
                                     "dev_ids": dev_ids, "timestamp": datetime.now().isoformat()},
                        "samples": results,
                    }, f, indent=2)
                print(f"Saved {len(results)} results to {out_path}")
            spreads[base_tag] = accuracies

    for tag, results in all_results.items():
        summaries[tag] = report(tag, results)

    print(f"\n{'=' * 60}\n  GRID SUMMARY\n{'=' * 60}")
    print(f"  {'Config':<28} {'Accuracy':>10} {'Contra. Recall':>16} {'Dropped-by-selector':>22}")
    print("  " + "-" * 78)
    for tag, s in summaries.items():
        print(f"  {tag:<28} {s['accuracy']:>9.2%} {s['contradiction_recall']:>15.2%} "
              f"{s['dropped_by_selector_count']:>10}/{s['failure_count']}")

    if args.repeats > 1:
        print(f"\n{'=' * 60}\n  ACCURACY SPREAD ACROSS {args.repeats} REPEATS\n{'=' * 60}")
        for base_tag, accs in spreads.items():
            mean = sum(accs) / len(accs)
            print(f"  {base_tag:<28} runs={[f'{a:.2%}' for a in accs]}  "
                  f"mean={mean:.2%}  min={min(accs):.2%}  max={max(accs):.2%}  "
                  f"spread={max(accs) - min(accs):.2%}")


if __name__ == "__main__":
    main()
