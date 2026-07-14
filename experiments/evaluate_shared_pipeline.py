"""
experiments/evaluate_shared_pipeline.py

Shared toolbox that experiments/compare_question_methods.py* and
experiments/ablation_p_question_selectors.py both import from — they
cannot run without it:

  - pick_dev_set(n, split)  — pins the exact same N sample ids every
    run, from either the "test" or "dev" split, so different runs are
    directly comparable (and dedupes an id collision in the raw data).
  - precision_recall_f1(results) — computes per-label precision/
    recall/F1 from a list of {gold_label/label, prediction} dicts.
  - _gold(r) — small patch so zero_shot's differently-named "label"
    key doesn't crash comparisons against the other methods, which use
    "gold_label".

(*compare_question_methods.py currently lives in the repo root, not
here — it imports this file as `from experiments.evaluate_shared_pipeline
import ...`.)

This file used to also have its own standalone script (a `main()` that ran
q2_pipeline/p_question/h_question/h_multihop on a fixed dev set). That's
gone now — q2_pipeline and h_multihop were retired to
experiments/archived_methods/, and compare_question_methods.py already
covers the current, more useful method set (zero_shot/retrieve_then_classify/
p_question/h_question/bridge_question) better than this file's old runner
did. Only the three helpers above remain.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the imports below

from data.data import load_split

LABELS = ["Entailment", "Contradiction", "Neutral"]


def pick_dev_set(n: int, split: str = "test") -> list[dict]:
    """Pin a fixed, reproducible slice of a held-out split — same ids every
    run for a given (split, n).

    Held-out data, not load_data()'s train+test concatenation: main.py's
    load_data() merges both splits for full-dataset experiment runs, but a
    dev set used to compare pipeline variants shouldn't be drawn from train.
    `split="dev"` is a third, completely separate split (799 samples) never
    touched by load_data() at all — use it for a clean, never-inspected
    number after `split="test"` has been used for iteration (id-level
    failures looked at, prompts patched against them).
    The test split has one duplicate "id_<n>" — dedupe, keeping the first
    occurrence, so the pinned set is n distinct samples.
    """
    samples = load_split(split)
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
    # the unified methods emit "gold_label".
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
