"""
compare_question_methods.py — Compare zero_shot (no evidence-finding at all),
retrieve_then_classify (Locator only, no questions), p_question (premise-
blind), h_question (hypothesis-blind), and bridge_question (both-texts) on
the same fixed held-out test-split dev set.

    zero_shot → retrieve_then_classify gap  = value of the Locator alone.
    retrieve_then_classify → question methods gap = value ADDED by
        generating questions on top of just locating. This is the core
        research claim behind the whole question-based-method family.

Reports, per method: accuracy + per-label precision/recall/F1.

For bridge_question specifically, also reports:
  - no_bridge_generated rate (how often Stage 1 failed to produce a
    bridging/comparison question even after the retry).
  - For every Contradiction/Entailment sample that collapsed to Neutral,
    which subset of its answered questions (bridge-tagged only, vs
    normal-tagged only) actually carries the decisive evidence — i.e.
    re-classifying with just that subset, does it recover the gold label?
    This tells us whether the bridging questions are doing the multi-hop
    work or whether the normal questions carry it regardless.

Because the lab backend has shown real run-to-run variance on an identical
config, the winning method (by accuracy on the first pass) is re-run twice
more and its accuracy spread is reported, not a single number.

Usage:
    python compare_question_methods.py --model qwen2.5-32b [--n 20]
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the imports below

from config.config import DEFAULT_PARAMS, RESULTS_DIR, setup_logger
from experiments.evaluate_shared_pipeline import pick_dev_set, precision_recall_f1, _gold
from prompts.shared_classifier import classify_evidence
from runner import zero_shot, retrieve_then_classify, p_question, h_question, bridge_question

METHODS = {
    "zero_shot":              zero_shot,
    "retrieve_then_classify": retrieve_then_classify,
    "p_question":             p_question,
    "h_question":             h_question,
    "bridge_question":        bridge_question,
}

FAILURE_GOLD_LABELS = ("Entailment", "Contradiction")


def diagnose_bridge_sample(result: dict, model: str, params: dict) -> str:
    """For a bridge_question sample that collapsed to Neutral, check which
    subset of its answered qa_pairs (bridge-tagged vs normal-tagged) alone
    would have recovered the gold label."""
    hypothesis = result["hypothesis"]
    gold       = _gold(result)
    qa_pairs   = result["qa_pairs"]

    bridge_pairs = [qa for qa in qa_pairs if qa["is_bridge"]]
    normal_pairs = [qa for qa in qa_pairs if not qa["is_bridge"]]

    bridge_pred = classify_evidence(model, params, bridge_pairs, hypothesis) if bridge_pairs else None
    normal_pred = classify_evidence(model, params, normal_pairs, hypothesis) if normal_pairs else None

    bridge_carries = bridge_pred == gold
    normal_carries = normal_pred == gold

    if bridge_carries and not normal_carries:
        return "bridge_carries_decisive_evidence"
    if normal_carries and not bridge_carries:
        return "normal_carries_decisive_evidence"
    if bridge_carries and normal_carries:
        return "both_subsets_sufficient"
    return "neither_subset_sufficient"


def run_method(name: str, dev_set: list[dict], model: str, params: dict) -> list[dict]:
    print(f"\n{'=' * 60}\n  Running {name}\n{'=' * 60}")
    results = METHODS[name].run(dev_set, model=model, params=params)

    if name == "bridge_question":
        for r in results:
            if _gold(r) in FAILURE_GOLD_LABELS and r["prediction"] == "Neutral" and r["qa_pairs"]:
                r["bridge_diagnosis"] = diagnose_bridge_sample(r, model, params)

    return results


def report(name: str, results: list[dict]) -> dict:
    n = len(results)
    correct = sum(_gold(r) == r["prediction"] for r in results)
    accuracy = correct / n if n else 0.0

    print(f"\n{'=' * 60}\n  {name}  ({n} samples)\n{'=' * 60}")
    print(f"  Accuracy: {correct}/{n} = {accuracy:.2%}")

    stats = precision_recall_f1(results)
    print(f"\n  {'Label':<16} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 54)
    for lbl in ["Entailment", "Contradiction", "Neutral"]:
        s = stats[lbl]
        print(f"  {lbl:<16} {s['support']:>8} {s['precision']:>9.2%} {s['recall']:>7.2%} {s['f1']:>7.2%}")

    if name == "bridge_question":
        no_bridge = sum(1 for r in results if r["no_bridge_generated"])
        print(f"\n  no_bridge_generated: {no_bridge}/{n}")

        diagnosed = [r for r in results if "bridge_diagnosis" in r]
        if diagnosed:
            print(f"  Gold Entailment/Contradiction predicted Neutral (with answerable evidence): {len(diagnosed)}")
            for r in diagnosed:
                print(f"    id={r['id']}  gold={_gold(r)}  diagnosis={r['bridge_diagnosis']}")
    print()

    return {"accuracy": accuracy, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--split", choices=["test", "dev"], default="test",
                         help="Which held-out split to draw from — 'dev' is untouched by any prior iteration")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--spread_method", choices=list(METHODS), default=None,
                         help="Force the 3x spread check onto this method instead of the accuracy winner")
    parser.add_argument("--no_spread", action="store_true",
                         help="Skip the spread check entirely (it re-runs a method 2 more times)")
    args = parser.parse_args()

    setup_logger("compare_question_methods", args.model)

    params = dict(DEFAULT_PARAMS)
    if args.max_tokens is not None:
        params["max_tokens"] = args.max_tokens

    dev_set = pick_dev_set(args.n, split=args.split)
    dev_ids = [s["id"] for s in dev_set]
    print(f"Pinned {args.split} set ({len(dev_ids)} ids): {dev_ids[:10]}{'...' if len(dev_ids) > 10 else ''}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "-")

    summaries = {}
    for name in args.methods:
        results = run_method(name, dev_set, args.model, params)
        out_path = os.path.join(RESULTS_DIR, f"{name}_{safe_model}_{args.split}{args.n}_{timestamp}.json")
        with open(out_path, "w") as f:
            json.dump({
                "metadata": {"experiment": name, "model": args.model, "split": args.split,
                             "dev_ids": dev_ids, "timestamp": datetime.now().isoformat()},
                "samples": results,
            }, f, indent=2)
        print(f"Saved {len(results)} results to {out_path}")
        summaries[name] = report(name, results)

    print(f"\n{'=' * 60}\n  COMPARISON SUMMARY\n{'=' * 60}")
    print(f"  {'Method':<20} {'Accuracy':>10}")
    print("  " + "-" * 32)
    for name, s in summaries.items():
        print(f"  {name:<20} {s['accuracy']:>9.2%}")

    # Re-run a config twice more; report spread instead of a single number
    # (the lab backend has shown ~25-point run-to-run variance on an
    # identical config). Defaults to the accuracy winner; --spread_method
    # forces a specific one (e.g. retrieve_then_classify, the control).
    if args.no_spread:
        spread_target = None
    elif args.spread_method:
        spread_target = args.spread_method
    elif len(summaries) > 1:
        spread_target = max(summaries, key=lambda k: summaries[k]["accuracy"])
    else:
        spread_target = None

    if spread_target:
        base_acc = summaries[spread_target]["accuracy"] if spread_target in summaries else None
        print(f"\n{'=' * 60}\n  SPREAD CHECK — {spread_target} (3 total runs)\n{'=' * 60}")
        accuracies = [base_acc] if base_acc is not None else []
        n_more = 3 - len(accuracies)
        for rep in range(n_more):
            results = METHODS[spread_target].run(dev_set, model=args.model, params=params)
            acc = sum(_gold(r) == r["prediction"] for r in results) / len(results)
            accuracies.append(acc)
            print(f"  rep {len(accuracies)}: {acc:.2%}")
        mean = sum(accuracies) / len(accuracies)
        print(f"  runs={[f'{a:.2%}' for a in accuracies]}  mean={mean:.2%}  "
              f"min={min(accuracies):.2%}  max={max(accuracies):.2%}  spread={max(accuracies)-min(accuracies):.2%}")


if __name__ == "__main__":
    main()
