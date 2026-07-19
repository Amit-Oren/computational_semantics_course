"""
experiments/ablation_full.py — Full ablation sweep over seeder × aggregation × method.

Compares all question-pipeline methods across the three orthogonal ablation
dimensions from the project design:

  Dimension 1 — Input processing (seeder):
    pos  : POS keyphrases (NLTK, deterministic)
    svo  : Dependency SVO triples (spaCy)
    srl  : Semantic Role Labeling (LLM-based, Pyatkin 2021 grounding)

  Dimension 2 — Classification aggregation:
    aggregated     : all Q/A pairs → one classifier call (QAGS/SummaC paradigm)
    sequential_cot : one Q/A at a time, running verdict (Decomposed Prompting)
    voting         : one call per Q/A pair, majority vote

  Dimension 3 — Method (question source):
    h_question : probe questions from H (keyphrases seed the LLM)
    p_question : questions from P (decomposition / freeform / seeded)

p_question also has two seeder-free generation modes (decomposition, freeform)
that run as additional baselines with each aggregation mode.

Results are saved per-config to results/ as JSON files and a summary TSV.

Usage (from repo root):
    python experiments/ablation_full.py --model qwen2.5-32b --n 30

Options:
    --model         Model name from config (default: qwen2.5-32b)
    --n             Number of samples from dev set (default: 30)
    --split         Data split to use: test or dev (default: test)
    --methods       Subset of methods: h_question p_question (default: both)
    --seeders       Subset of seeders: pos svo srl (default: pos srl; svo needs spaCy)
    --aggregations  Subset of aggregation modes (default: all three)
    --generations   p_question generation modes to include: decomposition freeform seeded
                    (default: decomposition seeded — freeform is the old baseline)
    --top_k         Stage 1b top-K for p_question (default: 3)
    --diagnose      Also run Neutral-failure diagnosis for p_question (slower)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DEFAULT_PARAMS, RESULTS_DIR, setup_logger
from experiments.evaluate_shared_pipeline import pick_dev_set, precision_recall_f1
from runner import bridge_question as bq_runner
from runner import few_shot_cot as fsc_runner
from runner import h_question as hq_runner
from runner import p_question as pq_runner
from runner import zero_shot as zs_runner
from utils.aggregation import AGGREGATION_MODES
from utils.seeding import SEEDERS

LABELS = ["Entailment", "Contradiction", "Neutral"]


# ── Metrics ───────────────────────────────────────────────────────────────────

def report_config(tag: str, results: list[dict]) -> dict:
    """Print per-label P/R/F1 + accuracy and return a summary dict."""
    n = len(results)
    if n == 0:
        print(f"\n  [{tag}] — no results")
        return {"tag": tag, "n": 0, "accuracy": 0.0}

    correct = sum(r["gold_label"] == r["prediction"] for r in results)
    accuracy = correct / n

    print(f"\n{'=' * 70}")
    print(f"  {tag}")
    print(f"  Accuracy: {correct}/{n} = {accuracy:.2%}")
    if "n_fact" in results[0]:
        avg_f = sum(r.get("n_fact", 0) for r in results) / n
        avg_r = sum(r.get("n_relation", 0) for r in results) / n
        print(f"  Avg questions: {avg_f:.1f} fact + {avg_r:.1f} relation")
    print(f"\n  {'Label':<16} {'N':>6} {'P':>8} {'R':>8} {'F1':>8}")
    print("  " + "-" * 50)

    stats = precision_recall_f1(results)
    row = {"tag": tag, "n": n, "accuracy": accuracy}
    for lbl in LABELS:
        s = stats[lbl]
        print(f"  {lbl:<16} {s['support']:>6} {s['precision']:>8.3f} {s['recall']:>8.3f} {s['f1']:>8.3f}")
        row[f"{lbl}_P"] = s["precision"]
        row[f"{lbl}_R"] = s["recall"]
        row[f"{lbl}_F1"] = s["f1"]

    # Macro F1
    macro_f1 = sum(stats[l]["f1"] for l in LABELS) / 3
    print(f"  {'Macro F1':<16} {'':>6} {'':>8} {'':>8} {macro_f1:>8.3f}")
    row["macro_f1"] = macro_f1

    # Neutral-failure rate (gold E/C but predicted N)
    n_fail = sum(
        1 for r in results
        if r["gold_label"] in ("Entailment", "Contradiction") and r["prediction"] == "Neutral"
    )
    print(f"  Neutral-fail (E/C→N): {n_fail}/{n}")
    row["neutral_fail_count"] = n_fail

    return row


def _safe_tag(tag: str) -> str:
    return tag.replace(" ", "_").replace("=", "-").replace("/", "-")


def find_existing_results(tag: str, model: str, results_dir: str) -> Optional[list[dict]]:
    """Return saved results for this tag+model if they exist, else None."""
    prefix = f"ablation_{_safe_tag(tag)}_{_safe_tag(model)}_"
    if not os.path.isdir(results_dir):
        return None
    matches = sorted(
        [f for f in os.listdir(results_dir) if f.startswith(prefix) and f.endswith(".json")],
        reverse=True,  # most recent first
    )
    if not matches:
        return None
    path = os.path.join(results_dir, matches[0])
    with open(path) as f:
        return json.load(f)


def save_results(tag: str, model: str, results: list[dict], results_dir: str) -> str:
    """Save results list to a JSON file in results_dir. Returns the path."""
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"ablation_{_safe_tag(tag)}_{_safe_tag(model)}_{ts}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def save_summary(rows: list[dict], results_dir: str) -> str:
    """Save a summary TSV and print a final comparison table."""
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"ablation_summary_{ts}.tsv")

    headers = ["tag", "n", "accuracy", "macro_f1",
               "Entailment_R", "Contradiction_R", "Neutral_R", "neutral_fail_count"]
    with open(path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY (sorted by accuracy desc)")
    print(f"{'=' * 70}")
    print(f"  {'Config':<45} {'Acc':>7} {'MacF1':>7} {'ContR':>7} {'N-fail':>7}")
    print("  " + "-" * 70)
    for row in sorted(rows, key=lambda r: r.get("accuracy", 0), reverse=True):
        print(
            f"  {row['tag'][:45]:<45} "
            f"{row.get('accuracy', 0):>7.3f} "
            f"{row.get('macro_f1', 0):>7.3f} "
            f"{row.get('Contradiction_R', 0):>7.3f} "
            f"{row.get('neutral_fail_count', 0):>7}"
        )
    return path


# ── Config runners ────────────────────────────────────────────────────────────

def _normalize_gold(results: list[dict]) -> list[dict]:
    """Rename 'label' → 'gold_label' for runners that use the old key."""
    for r in results:
        if "gold_label" not in r and "label" in r:
            r["gold_label"] = r.pop("label")
    return results


def run_zero_shot_config(dev_set: list[dict], model: str, params: dict) -> list[dict]:
    tag = "zero_shot"
    print(f"\n{'─' * 70}\n  {tag}\n{'─' * 70}")
    try:
        results = zs_runner.run(dev_set, model=model, params=params)
        return _normalize_gold(results)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return []


def run_few_shot_cot_config(dev_set: list[dict], model: str, params: dict) -> list[dict]:
    tag = "few_shot_cot"
    print(f"\n{'─' * 70}\n  {tag}\n{'─' * 70}")
    try:
        results = fsc_runner.run(dev_set, model=model, params=params)
        return _normalize_gold(results)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return []


def run_bridge_question_config(
    aggregation: str,
    dev_set: list[dict], model: str, params: dict,
) -> list[dict]:
    tag = f"bridge_question | agg={aggregation}"
    print(f"\n{'─' * 70}\n  {tag}\n{'─' * 70}")
    try:
        return bq_runner.run(
            dev_set, model=model, params=params,
            aggregation=aggregation,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return []


def run_h_question_config(
    seeder: str, aggregation: str,
    dev_set: list[dict], model: str, params: dict,
) -> list[dict]:
    tag = f"h_question | seeder={seeder} | agg={aggregation}"
    print(f"\n{'─' * 70}\n  {tag}\n{'─' * 70}")
    try:
        return hq_runner.run(
            dev_set, model=model, params=params,
            seeder_name=seeder, aggregation=aggregation,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return []


def run_p_question_config(
    generation: str, seeder: str, aggregation: str,
    dev_set: list[dict], model: str, params: dict,
    top_k: int,
) -> list[dict]:
    seeder_label = f"|seeder={seeder}" if generation == "seeded" else ""
    tag = f"p_question | gen={generation}{seeder_label} | agg={aggregation}"
    print(f"\n{'─' * 70}\n  {tag}\n{'─' * 70}")
    try:
        return pq_runner.run(
            dev_set, model=model, params=params,
            generation=generation,
            seeder_name=seeder if generation == "seeded" else "pos",
            aggregation=aggregation,
            top_k=top_k,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Full ablation sweep: seeder × aggregation × method")
    parser.add_argument("--model",        default="qwen2.5-32b")
    parser.add_argument("--n",            type=int, default=30)
    parser.add_argument("--split",        default="test", choices=["test", "dev"])
    parser.add_argument("--methods",      nargs="+",
                        default=["zero_shot", "few_shot_cot", "bridge_question", "h_question", "p_question"],
                        choices=["zero_shot", "few_shot_cot", "h_question", "p_question", "bridge_question"])
    parser.add_argument("--seeders",      nargs="+", default=["pos", "srl"],
                        choices=sorted(SEEDERS.keys()))
    parser.add_argument("--aggregations", nargs="+", default=list(AGGREGATION_MODES),
                        choices=list(AGGREGATION_MODES))
    parser.add_argument("--generations",  nargs="+", default=["decomposition", "seeded"],
                        choices=["decomposition", "freeform", "seeded"],
                        help="p_question generation modes to include")
    parser.add_argument("--top_k",        type=int, default=3)
    parser.add_argument("--diagnose",     action="store_true",
                        help="Run Neutral-failure diagnosis for p_question (slower)")
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    results_dir = RESULTS_DIR
    logger = setup_logger("ablation_full", args.model)

    print(f"\n{'=' * 70}")
    print("  ABLATION FULL SWEEP")
    print(f"  Model       : {args.model}")
    print(f"  Split       : {args.split}  |  N = {args.n}")
    print(f"  Methods     : {args.methods}")
    print(f"  Seeders     : {args.seeders}")
    print(f"  Aggregations: {args.aggregations}")
    print(f"  Generations : {args.generations}  (p_question only; bridge_question has no seeder)")
    print(f"{'=' * 70}")

    dev_set = pick_dev_set(args.n, args.split)
    print(f"  Loaded {len(dev_set)} samples from '{args.split}' split.\n")

    all_rows: list[dict] = []

    def _run_config(tag: str, runner_fn, *runner_args) -> None:
        """Run a config, skipping if results already saved for this tag+model."""
        existing = find_existing_results(tag, args.model, results_dir)
        if existing is not None:
            print(f"\n  [SKIP] {tag}  (loaded {len(existing)} saved results)")
            all_rows.append(report_config(tag, existing))
            return
        results = runner_fn(*runner_args)
        if results:
            save_results(tag, args.model, results, results_dir)
            all_rows.append(report_config(tag, results))

    # ── Baselines ─────────────────────────────────────────────────────────────
    if "zero_shot" in args.methods:
        _run_config("zero_shot", run_zero_shot_config, dev_set, args.model, params)

    if "few_shot_cot" in args.methods:
        _run_config("few_shot_cot", run_few_shot_cot_config, dev_set, args.model, params)

    # ── bridge_question configs ───────────────────────────────────────────────
    if "bridge_question" in args.methods:
        for agg in args.aggregations:
            tag = f"bridge_question | agg={agg}"
            _run_config(tag, run_bridge_question_config, agg, dev_set, args.model, params)

    # ── h_question configs ────────────────────────────────────────────────────
    if "h_question" in args.methods:
        for seeder in args.seeders:
            for agg in args.aggregations:
                tag = f"h_question | seeder={seeder} | agg={agg}"
                _run_config(tag, run_h_question_config, seeder, agg, dev_set, args.model, params)

    # ── p_question configs ────────────────────────────────────────────────────
    if "p_question" in args.methods:
        for generation in args.generations:
            if generation == "seeded":
                seeder_list = args.seeders
            else:
                seeder_list = ["pos"]  # dummy — not used when generation != "seeded"

            for seeder in seeder_list:
                for agg in args.aggregations:
                    seeder_label = f"|seeder={seeder}" if generation == "seeded" else ""
                    tag = f"p_question | gen={generation}{seeder_label} | agg={agg}"
                    _run_config(
                        tag, run_p_question_config,
                        generation, seeder, agg,
                        dev_set, args.model, params, args.top_k,
                    )

    # ── Final summary ─────────────────────────────────────────────────────────
    if all_rows:
        summary_path = save_summary(all_rows, results_dir)
        print(f"\n  Summary saved: {summary_path}")
    else:
        print("\n  No results to summarize.")


if __name__ == "__main__":
    main()
