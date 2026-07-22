"""
compare_results.py — Compare prediction accuracy across experiment result files.

Usage:
    # Compare two specific files
    python compare_results.py results/h_question_qwen2.5-32b.json results/bridge_question_qwen2.5-32b.json

    # Auto-discover and compare all files in results/
    python compare_results.py
"""

import json
import sys
import os
import glob
from collections import defaultdict

LABELS = ["Entailment", "Contradiction", "Neutral"]


def load(path: str) -> tuple[dict, list[dict]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("metadata", {}), data.get("samples", [])


def accuracy(samples: list[dict]) -> float:
    if not samples:
        return 0.0
    return sum(s["label"] == s["prediction"] for s in samples) / len(samples)


def per_label_stats(samples: list[dict]) -> dict:
    """Per-label accuracy, support, and confusion counts."""
    stats = {}
    for lbl in LABELS:
        gold_positives = [s for s in samples if s["label"] == lbl]
        correct = sum(s["prediction"] == lbl for s in gold_positives)
        predicted_as = defaultdict(int)
        for s in gold_positives:
            predicted_as[s["prediction"]] += 1
        stats[lbl] = {
            "support":  len(gold_positives),
            "correct":  correct,
            "accuracy": correct / len(gold_positives) if gold_positives else 0.0,
            "predicted_as": dict(predicted_as),
        }
    return stats


def confusion_matrix(samples: list[dict]) -> dict[str, dict[str, int]]:
    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for s in samples:
        g, p = s["label"], s["prediction"]
        if g in matrix and p in LABELS:
            matrix[g][p] += 1
    return matrix


def print_separator(char="─", width=72):
    print(char * width)


def print_single_report(tag: str, meta: dict, samples: list[dict]):
    print_separator("═")
    print(f"  {tag}")
    if meta:
        print(f"  Model: {meta.get('model', '?')}  |  Temp: {meta.get('temperature', '?')}  |  "
              f"Tokens: {meta.get('max_tokens', '?')}  |  Run: {meta.get('timestamp', '?')[:16]}")
    print_separator()

    n = len(samples)
    acc = accuracy(samples)
    correct = sum(s["label"] == s["prediction"] for s in samples)
    print(f"  Overall Accuracy : {correct}/{n} = {acc:.2%}")
    print()

    # Per-label breakdown
    stats = per_label_stats(samples)
    print(f"  {'Label':<16} {'Support':>8} {'Correct':>8} {'Accuracy':>10}  Predicted as")
    print_separator("·")
    for lbl in LABELS:
        st = stats[lbl]
        dist = "  ".join(f"{k}:{v}" for k, v in sorted(st["predicted_as"].items()))
        print(f"  {lbl:<16} {st['support']:>8} {st['correct']:>8} {st['accuracy']:>9.2%}  {dist}")
    print()

    # Confusion matrix
    cm = confusion_matrix(samples)
    col_w = 14
    print(f"  {'Confusion matrix (rows=gold, cols=pred)'}")
    print(f"  {'Gold \\ Pred':<16}" + "".join(f"{p:>{col_w}}" for p in LABELS))
    print_separator("·")
    for g in LABELS:
        row = "".join(f"{cm[g][p]:>{col_w}}" for p in LABELS)
        print(f"  {g:<16}{row}")
    print()


def print_comparison(tag_a: str, samples_a: list[dict], tag_b: str, samples_b: list[dict]):
    """Side-by-side accuracy delta and disagreement analysis."""
    # Index by id for alignment
    by_id_a = {s["id"]: s for s in samples_a}
    by_id_b = {s["id"]: s for s in samples_b}
    common_ids = sorted(set(by_id_a) & set(by_id_b))

    if not common_ids:
        print("  No samples with matching IDs — cannot compute delta or disagreements.")
        return

    shared_a = [by_id_a[i] for i in common_ids]
    shared_b = [by_id_b[i] for i in common_ids]

    acc_a = accuracy(shared_a)
    acc_b = accuracy(shared_b)
    delta = acc_b - acc_a

    print_separator("═")
    name_a = tag_a.split(os.sep)[-1].replace(".json", "")
    name_b = tag_b.split(os.sep)[-1].replace(".json", "")
    print(f"  COMPARISON  ({len(common_ids)} shared samples)")
    print_separator()
    print(f"  {'Experiment':<40} {'Accuracy':>10}  {'Correct':>8}")
    print_separator("·")
    correct_a = sum(s["label"] == s["prediction"] for s in shared_a)
    correct_b = sum(s["label"] == s["prediction"] for s in shared_b)
    print(f"  {name_a:<40} {acc_a:>9.2%}  {correct_a:>8}/{len(common_ids)}")
    print(f"  {name_b:<40} {acc_b:>9.2%}  {correct_b:>8}/{len(common_ids)}")
    print_separator("·")
    direction = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
    print(f"  {'Delta (B − A)':<40} {direction} {abs(delta):>8.2%}")
    print()

    # Per-label delta
    stats_a = per_label_stats(shared_a)
    stats_b = per_label_stats(shared_b)
    print(f"  Per-label accuracy delta  ({name_b} vs {name_a})")
    print(f"  {'Label':<16} {'A':>8} {'B':>8} {'Delta':>9}")
    print_separator("·")
    for lbl in LABELS:
        a_acc = stats_a[lbl]["accuracy"]
        b_acc = stats_b[lbl]["accuracy"]
        d = b_acc - a_acc
        direction = "▲" if d > 0 else ("▼" if d < 0 else "=")
        print(f"  {lbl:<16} {a_acc:>7.2%}  {b_acc:>7.2%}  {direction} {abs(d):>7.2%}")
    print()

    # Disagreements: A right, B wrong / B right, A wrong
    a_right_b_wrong = [
        i for i in common_ids
        if by_id_a[i]["label"] == by_id_a[i]["prediction"]
        and by_id_b[i]["label"] != by_id_b[i]["prediction"]
    ]
    b_right_a_wrong = [
        i for i in common_ids
        if by_id_b[i]["label"] == by_id_b[i]["prediction"]
        and by_id_a[i]["label"] != by_id_a[i]["prediction"]
    ]
    both_wrong = [
        i for i in common_ids
        if by_id_a[i]["label"] != by_id_a[i]["prediction"]
        and by_id_b[i]["label"] != by_id_b[i]["prediction"]
    ]
    both_right = [
        i for i in common_ids
        if by_id_a[i]["label"] == by_id_a[i]["prediction"]
        and by_id_b[i]["label"] == by_id_b[i]["prediction"]
    ]

    print(f"  Agreement breakdown")
    print_separator("·")
    print(f"  Both correct          : {len(both_right):>5}  ({len(both_right)/len(common_ids):.2%})")
    print(f"  Both wrong            : {len(both_wrong):>5}  ({len(both_wrong)/len(common_ids):.2%})")
    print(f"  Only A correct        : {len(a_right_b_wrong):>5}  ({len(a_right_b_wrong)/len(common_ids):.2%})")
    print(f"  Only B correct        : {len(b_right_a_wrong):>5}  ({len(b_right_a_wrong)/len(common_ids):.2%})")
    print()

    # Show first few disagreements (B gained)
    if b_right_a_wrong:
        print(f"  Samples where {name_b} gained (first 5)")
        print_separator("·")
        for sid in b_right_a_wrong[:5]:
            sa, sb = by_id_a[sid], by_id_b[sid]
            print(f"  id={sid}  gold={sa['label']}  A_pred={sa['prediction']}  B_pred={sb['prediction']}")
        print()

    # Show first few where B lost
    if a_right_b_wrong:
        print(f"  Samples where {name_b} lost (first 5)")
        print_separator("·")
        for sid in a_right_b_wrong[:5]:
            sa, sb = by_id_a[sid], by_id_b[sid]
            print(f"  id={sid}  gold={sa['label']}  A_pred={sa['prediction']}  B_pred={sb['prediction']}")
        print()


def discover_files() -> list[str]:
    return sorted(glob.glob("results/*.json"))


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else discover_files()

    if not paths:
        print("No result files found. Run an experiment first:\n"
              "  python main.py --experiment h_question      --model <model>\n"
              "  python main.py --experiment bridge_question --model <model>")
        return

    # ── Individual reports ────────────────────────────────────────────────────
    loaded = []
    for path in paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        meta, samples = load(path)
        loaded.append((path, meta, samples))
        print_single_report(path, meta, samples)

    # ── Pairwise comparison (all pairs if >2, or just the pair if exactly 2) ─
    if len(loaded) >= 2:
        pairs = [(loaded[0], loaded[1])] if len(loaded) == 2 else [
            (loaded[i], loaded[j])
            for i in range(len(loaded))
            for j in range(i + 1, len(loaded))
        ]
        for (path_a, _, sa), (path_b, _, sb) in pairs:
            print_comparison(path_a, sa, path_b, sb)

    print_separator("═")


if __name__ == "__main__":
    main()
