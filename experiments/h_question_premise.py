"""
experiments/h_question_premise.py — Isolated test: does adding the full
premise to the classifier call improve h_question specifically?

Does NOT touch prompts/shared_classifier.py or any other method. This is a
standalone check with its own local classify-with-premise function, run only
through h_question's existing Stage 1 (question generation) and shared
Stage 2/3 (locate_and_answer, unchanged). If this shows a real improvement,
the shared classifier can be updated deliberately later — this script is
just the cheap check first.

Usage (from the repo root):
    python experiments/h_question_premise.py --model qwen2.5-32b --n 100 --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the imports below

from langchain_core.messages import SystemMessage, HumanMessage

from config.config import DEFAULT_PARAMS, RESULTS_DIR, ClassifyOutput, get_structured_llm, logger, setup_logger
from evaluate_shared_pipeline import pick_dev_set, precision_recall_f1, _gold
from prompts.shared_classifier import CLASSIFIER_SYSTEM_PROMPT, build_evidence_block, FALLBACK_LABEL
from runner.h_question import HQuestionPipeline
from runner import zero_shot
from utils.pos_keyphrase import extract_keyphrases
from utils.premise_indexer import number_sentences
from utils.locator_extractor import locate_and_answer
from utils.retry import call_with_retry

# Local-only prompt variant — premise added, not touching the shared one.
CLASSIFIER_USER_PROMPT_WITH_PREMISE = """\
PREMISE:
{premise}

EVIDENCE:
{evidence_block}

HYPOTHESIS: "{hypothesis}"

Classify the relationship between the evidence and the hypothesis.\
"""


def classify_evidence_with_premise(model: str, params: dict, premise: str, qa_pairs: list[dict], hypothesis: str) -> str:
    """Local-only variant of classify_evidence that also includes the full
    premise. Not part of the shared module — experimental, h_question only."""
    if not qa_pairs:
        return FALLBACK_LABEL
    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=CLASSIFIER_USER_PROMPT_WITH_PREMISE.format(
            premise=premise, evidence_block=build_evidence_block(qa_pairs), hypothesis=hypothesis,
        )),
    ]
    llm = get_structured_llm(model, ClassifyOutput, params)
    try:
        out = call_with_retry(llm.invoke, messages)
    except Exception as exc:
        logger.warning(f"classifier(+premise) call failed: {exc}")
        return FALLBACK_LABEL
    return out.label if out else FALLBACK_LABEL


def run_h_question_with_premise(samples: list[dict], model: str, params: dict) -> list[dict]:
    pipeline = HQuestionPipeline(model, params)
    results = []
    for i, sample in enumerate(samples):
        premise    = sample["premise"]
        hypothesis = sample["hypothesis"]
        warnings: list[str] = []

        kp_result  = extract_keyphrases(hypothesis)
        keyphrases = kp_result["keyphrases"]

        q_output = None
        try:
            q_output = pipeline.stage1_generate_questions(hypothesis, keyphrases)
        except Exception as exc:
            logger.warning(f"Stage 1 failed: {exc}")

        if q_output is None or not q_output.questions:
            results.append({
                "id": sample.get("id"), "gold_label": sample["label"], "prediction": FALLBACK_LABEL,
                "qa_pairs": [], "warnings": ["Stage 1 returned no questions."],
            })
            logger.info(f"[{i+1}/{len(samples)}] id={sample.get('id')} | gold={sample['label']} | pred={FALLBACK_LABEL} (no questions)")
            continue

        indexed_sentences, numbered_premise = number_sentences(premise)
        qa_pairs = []
        for question in q_output.questions:
            r = locate_and_answer(
                model, params, question,
                indexed_sentences=indexed_sentences, numbered_premise=numbered_premise,
            )
            if r["answerable"]:
                qa_pairs.append(r)

        if not qa_pairs:
            prediction = FALLBACK_LABEL
            warnings.append("All questions unanswerable.")
        else:
            prediction = classify_evidence_with_premise(model, params, premise, qa_pairs, hypothesis)

        results.append({
            "id": sample.get("id"), "gold_label": sample["label"], "prediction": prediction,
            "qa_pairs": qa_pairs, "warnings": warnings,
        })
        logger.info(f"[{i+1}/{len(samples)}] id={sample.get('id')} | gold={sample['label']} | pred={prediction}")

    return results


def report(name: str, results: list[dict]) -> float:
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
    print()
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--split", choices=["test", "dev"], default="dev")
    parser.add_argument("--max_tokens", type=int, default=None)
    args = parser.parse_args()

    setup_logger("experiment_h_question_premise", args.model)
    params = dict(DEFAULT_PARAMS)
    if args.max_tokens is not None:
        params["max_tokens"] = args.max_tokens

    dev_set = pick_dev_set(args.n, split=args.split)
    dev_ids = [s["id"] for s in dev_set]
    print(f"Pinned {args.split} set ({len(dev_ids)} ids): {dev_ids[:10]}...")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "-")

    print(f"\n{'=' * 60}\n  Running h_question_with_premise (experimental)\n{'=' * 60}")
    hq_premise_results = run_h_question_with_premise(dev_set, args.model, params)
    with open(os.path.join(RESULTS_DIR, f"h_question_with_premise_{safe_model}_{args.split}{args.n}_{timestamp}.json"), "w") as f:
        json.dump({"metadata": {"experiment": "h_question_with_premise", "model": args.model,
                                 "split": args.split, "dev_ids": dev_ids}, "samples": hq_premise_results}, f, indent=2)

    print(f"\n{'=' * 60}\n  Running zero_shot (baseline for comparison)\n{'=' * 60}")
    zs_results = zero_shot.run(dev_set, model=args.model, params=params)
    with open(os.path.join(RESULTS_DIR, f"zero_shot_{safe_model}_{args.split}{args.n}_{timestamp}.json"), "w") as f:
        json.dump({"metadata": {"experiment": "zero_shot", "model": args.model,
                                 "split": args.split, "dev_ids": dev_ids}, "samples": zs_results}, f, indent=2)

    acc_premise = report("h_question_with_premise", hq_premise_results)
    acc_zs = report("zero_shot", zs_results)

    print(f"{'=' * 60}\n  SUMMARY\n{'=' * 60}")
    print(f"  h_question_with_premise : {acc_premise:.2%}")
    print(f"  zero_shot               : {acc_zs:.2%}")
    print(f"  delta                   : {acc_premise - acc_zs:+.2%}")


if __name__ == "__main__":
    main()
