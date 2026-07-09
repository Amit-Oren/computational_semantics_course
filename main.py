import json
import os
import argparse
from datetime import datetime
from config.config import (
    MODELS, RESULTS_DIR, DEFAULT_PARAMS,
    P_QUESTION_SELECTOR, P_QUESTION_SELECTION, P_QUESTION_TOP_K, P_QUESTION_MMR_LAMBDA,
    P_QUESTION_MAX_QUESTIONS,
    setup_logger,
)
from utils.question_selectors import SELECTORS, SELECTION_MODES
from data.data import load_data
from runner import zero_shot, few_shot_cot, hdqd_pipeline, q2_pipeline, p_question, h_question, h_multihop

RUNNERS = {
    "zero_shot":     zero_shot,
    "few_shot_cot":  few_shot_cot,
    "hdqd_pipeline": hdqd_pipeline,
    "q2_pipeline":   q2_pipeline,
    "p_question":    p_question,
    "h_question":    h_question,
    "h_multihop":    h_multihop,
}


def save_results(results: list[dict], experiment: str, model: str, params: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "-")
    out_path = os.path.join(RESULTS_DIR, f"{experiment}_{safe_model}_{timestamp}.json")
    output = {
        "metadata": {
            "experiment": experiment,
            "model": model,
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "seed": params.get("seed", None),
            "timestamp": datetime.now().isoformat(),
        },
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=RUNNERS.keys(), required=True)
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N samples")
    parser.add_argument("--max_tokens", type=int, default=None, help="Override max tokens per LLM call")
    parser.add_argument("--selector", choices=SELECTORS, default=P_QUESTION_SELECTOR,
                         help="Stage 1b relevance scorer (p_question only)")
    parser.add_argument("--selection", choices=SELECTION_MODES, default=P_QUESTION_SELECTION,
                         help="Stage 1b selection mode: topk or mmr (p_question only)")
    parser.add_argument("--top_k", type=int, default=P_QUESTION_TOP_K,
                         help="Number of questions Stage 1b keeps (p_question only)")
    parser.add_argument("--mmr_lambda", type=float, default=P_QUESTION_MMR_LAMBDA,
                         help="MMR relevance/diversity balance, 1.0 = pure top-K (p_question only)")
    parser.add_argument("--max_questions", type=int, default=P_QUESTION_MAX_QUESTIONS,
                         help="Cap on Stage 1a decomposed questions, relations kept first (p_question only)")
    args = parser.parse_args()

    setup_logger(args.experiment, args.model)

    params = dict(DEFAULT_PARAMS)
    if args.max_tokens is not None:
        params["max_tokens"] = args.max_tokens

    samples = load_data()
    if args.limit:
        samples = samples[:args.limit]
    runner = RUNNERS[args.experiment]
    if args.experiment == "p_question":
        results = runner.run(
            samples, model=args.model, params=params,
            selector=args.selector, selection=args.selection,
            top_k=args.top_k, mmr_lambda=args.mmr_lambda,
            max_questions=args.max_questions,
        )
    else:
        results = runner.run(samples, model=args.model, params=params)
    save_results(results, args.experiment, args.model, params)


if __name__ == "__main__":
    main()
