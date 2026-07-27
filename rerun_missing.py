"""Re-run only the samples missing from an existing production result file and append them."""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import DEFAULT_PARAMS
from data.data import load_split

MODEL  = "google/gemma-3-27b-it"
PARAMS = {"temperature": 0.0, "max_tokens": 4096}

TARGETS = {
    "zero_shot": {
        "file":    "results/production/production_train_zero_shot_google-gemma-3-27b-it_20260723_131218.json",
        "missing": ["id_5584", "id_66"],
    },
    "few_shot_cot": {
        "file":    "results/production/production_train_few_shot_cot_google-gemma-3-27b-it_20260723_131615.json",
        "missing": ["id_762"],
    },
}


def run_zero_shot(sample: dict) -> dict | None:
    from prompts.zero_shot import SYSTEM_PROMPT, USER_PROMPT
    from runner.zero_shot import call
    user = USER_PROMPT.format(premise=sample["premise"], hypothesis=sample["hypothesis"])
    output = call(SYSTEM_PROMPT, user, MODEL, PARAMS)
    if output is None:
        return None
    return {"id": sample["id"], "label": sample["label"],
            "prediction": output.label, "explanation": output.explanation}


def run_few_shot_cot(sample: dict, examples: str) -> dict | None:
    from prompts.few_shot_cot import SYSTEM_PROMPT, USER_PROMPT
    from runner.few_shot_cot import call
    user = USER_PROMPT.format(examples=examples, premise=sample["premise"], hypothesis=sample["hypothesis"])
    output = call(SYSTEM_PROMPT, user, MODEL, PARAMS)
    if output is None:
        return None
    return {"id": sample["id"], "label": sample["label"],
            "prediction": output.label, "explanation": output.explanation}


def main():
    train = {s["id"]: s for s in load_split("train")}

    # few_shot_cot needs examples pre-built
    from runner.few_shot_cot import select_shots
    from prompts.few_shot_cot import format_examples
    examples = format_examples(select_shots())

    for method, cfg in TARGETS.items():
        path    = cfg["file"]
        missing = cfg["missing"]
        print(f"\n── {method}: re-running {missing}")

        with open(path) as f:
            results = json.load(f)

        for sid in missing:
            sample = train[sid]
            try:
                if method == "zero_shot":
                    result = run_zero_shot(sample)
                else:
                    result = run_few_shot_cot(sample, examples)

                if result:
                    results.append(result)
                    print(f"  {sid} → {result['prediction']} (gold={result['label']})")
                else:
                    print(f"  {sid} → FAILED again (None returned)")
            except Exception as e:
                print(f"  {sid} → ERROR: {e}")

        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved {len(results)} results to {path}")


if __name__ == "__main__":
    main()
