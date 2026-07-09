"""
evaluate_shared_pipeline.py — Run q2_pipeline, p_question, h_question, and
h_multihop on the same fixed 20-sample dev set and report accuracy + per-label
precision/recall/F1 for each, plus a flag for samples with empty qa_pairs.

These four methods share identical evidence-finding (locate_and_answer) and
classification (classify_evidence) components; this script exists because
they now emit "gold_label" (not "label") in their per-sample output, so
compare_results.py (built for the older schema) doesn't apply to them.

Usage:
    python evaluate_shared_pipeline.py --model <model> [--n 20]
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

from config.config import DEFAULT_PARAMS, RESULTS_DIR
from data.data import load_split
from runner import q2_pipeline, p_question, h_question, h_multihop, zero_shot

LABELS = ["Entailment", "Contradiction", "Neutral"]

RUNNERS = {
    "q2_pipeline": q2_pipeline,
    "p_question":  p_question,
    "h_question":  h_question,
    "h_multihop":  h_multihop,
    "zero_shot":   zero_shot,
}


def pick_dev_set(n: int) -> list[dict]:
    """Pin a fixed, reproducible slice of the held-out test split — same ids
    every run.

    Held-out data, not load_data()'s train+test concatenation: main.py's
    load_data() merges both splits for full-dataset experiment runs, but a
    dev set used to compare pipeline variants shouldn't be drawn from train.
    The test split itself has one duplicate "id_<n>" — dedupe, keeping the
    first occurrence, so the pinned set is n distinct samples.
    """
    samples = load_split("test")
    samples = sorted(samples, key=lambda s: s["id"])
    seen: set[str] = set()
    deduped = []
    for s in samples:
        if s["id"] not in seen:
            seen.add(s["id"])
            deduped.append(s)
    return deduped[:n]


def _gold(r: dict) -> str:
    # zero_shot (untouched, out of refactor scope) still emits "label";
    # the four unified methods emit "gold_label".
    return r.get("gold_label", r.get("label"))


def precision_recall_f1(results: list[dict]) -> dict:
    stats = {}
    for lbl in LABELS:
        tp = sum(1 for r in results if r["prediction"] == lbl and _gold(r) == lbl)
        fp = sum(1 for r in results if r["prediction"] == lbl and _gold(r) != lbl)
        fn = sum(1 for r in results if r["prediction"] != lbl and _gold(r) == lbl)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        stats[lbl] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return stats


def report(method: str, results: list[dict]) -> None:
    n = len(results)
    correct = sum(_gold(r) == r["prediction"] for r in results)
    accuracy = correct / n if n else 0.0

    print("=" * 60)
    print(f"  {method}  ({n} samples)")
    print("=" * 60)
    print(f"  Accuracy: {correct}/{n} = {accuracy:.2%}")

    # qa_pairs only exists on the four unified methods, not on zero_shot.
    if results and "qa_pairs" in results[0]:
        empty_qa = [r for r in results if not r.get("qa_pairs")]
        if empty_qa:
            print(f"  Samples with empty qa_pairs (auto-Neutral): {len(empty_qa)}")
            for r in empty_qa:
                flag = "OK" if r["prediction"] == "Neutral" else "MISMATCH — not predicted Neutral!"
                print(f"    id={r['id']}  gold={_gold(r)}  pred={r['prediction']}  [{flag}]")

    stats = precision_recall_f1(results)
    print(f"\n  {'Label':<16} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 54)
    for lbl in LABELS:
        s = stats[lbl]
        print(f"  {lbl:<16} {s['support']:>8} {s['precision']:>9.2%} {s['recall']:>7.2%} {s['f1']:>7.2%}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=None)
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    if args.max_tokens is not None:
        params["max_tokens"] = args.max_tokens

    dev_set = pick_dev_set(args.n)
    dev_ids = [s["id"] for s in dev_set]
    print(f"Pinned dev set ({len(dev_ids)} ids): {dev_ids}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "-")

    all_results = {}
    for method, runner in RUNNERS.items():
        try:
            results = runner.run(dev_set, model=args.model, params=params)
        except Exception as exc:
            print(f"  {method} crashed — skipping this method: {exc}\n")
            continue
        all_results[method] = results

        out_path = os.path.join(RESULTS_DIR, f"{method}_{safe_model}_devset_{timestamp}.json")
        with open(out_path, "w") as f:
            json.dump({
                "metadata": {"experiment": method, "model": args.model, "dev_ids": dev_ids,
                             "timestamp": datetime.now().isoformat()},
                "samples": results,
            }, f, indent=2)
        print(f"Saved {len(results)} results to {out_path}\n")

    for method, results in all_results.items():
        report(method, results)


if __name__ == "__main__":
    main()
