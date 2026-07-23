"""
Production run — all 9 project methods on the ConTRoL train split, n=1500,
for a single model. Run once per model (qwen2.5-32b now; gpt-oss-20b and
gemma4-31b later once swapped in on the lab server).

Runs samples WITHIN each method concurrently (default 5 workers) — confirmed
via direct testing that the lab server genuinely parallelizes concurrent
requests to the same loaded model rather than queuing them serially, so
batch time is bounded by the slowest sample in a batch, not the sum.
Methods themselves still run one after another (not concurrently with each
other), since only one model is loaded on the GPU at a time.

Saves each method's full result list to results/ as soon as that method
finishes, so an interruption partway through doesn't lose already-completed
methods.

Usage: python run_production_train.py <model> [max_workers]
"""
from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import RESULTS_DIR, setup_logger, logger, DEFAULT_PARAMS
from data.data import load_split

PRODUCTION_RESULTS_DIR = os.path.join(RESULTS_DIR, "production")
PRODUCTION_SEED = 42
SAMPLE_IDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "production_train_sample_ids.json")


def get_production_samples(n: int) -> list[dict]:
    """Fixed random-seed sample from the train split, saved once so every
    model uses the exact same n samples (not a sorted-by-id slice, which
    string-sorts IDs like id_1, id_10, id_100, id_1000, id_1001, ..., id_11
    — an arbitrary, non-representative order, not numeric and not random)."""
    train = load_split("train")
    if os.path.exists(SAMPLE_IDS_PATH):
        with open(SAMPLE_IDS_PATH) as f:
            ids = json.load(f)
        by_id = {s["id"]: s for s in train}
        return [by_id[i] for i in ids]

    rng = random.Random(PRODUCTION_SEED)
    picked = rng.sample(train, n)
    with open(SAMPLE_IDS_PATH, "w") as f:
        json.dump([s["id"] for s in picked], f, indent=2)
    logger.info(f"Generated {len(picked)} fixed-seed (seed={PRODUCTION_SEED}) production sample IDs -> {SAMPLE_IDS_PATH}")
    return picked
from utils.retry import call_with_retry

N = 1500
PARAMS = {"temperature": 0.0, "max_tokens": 4096}
DEFAULT_MAX_WORKERS = 5


# ─── zero_shot / few_shot_cot: no pipeline class, wrap their call() helper ───

def _zero_shot_one(sample: dict, model: str, params: dict) -> dict | None:
    from prompts.zero_shot import SYSTEM_PROMPT, USER_PROMPT
    from runner.zero_shot import call
    user = USER_PROMPT.format(premise=sample["premise"], hypothesis=sample["hypothesis"])
    output = call(SYSTEM_PROMPT, user, model, params)
    if output is None:
        return None
    return {
        "id": sample.get("id"), "label": sample["label"],
        "prediction": output.label, "explanation": output.explanation,
    }


def _few_shot_cot_one(sample: dict, model: str, params: dict, examples: str) -> dict | None:
    from prompts.few_shot_cot import SYSTEM_PROMPT, USER_PROMPT
    from runner.few_shot_cot import call
    user = USER_PROMPT.format(examples=examples, premise=sample["premise"], hypothesis=sample["hypothesis"])
    output = call(SYSTEM_PROMPT, user, model, params)
    if output is None:
        return None
    return {
        "id": sample.get("id"), "label": sample["label"],
        "prediction": output.label, "explanation": output.explanation,
    }


# ─── Method registry: how to build the per-sample callable for each method ───

def _build_worker(label: str, model: str, params: dict):
    """Returns a callable(sample) -> dict|None for the given method."""
    if label == "zero_shot":
        return lambda sample: _zero_shot_one(sample, model, params)

    if label == "few_shot_cot":
        from runner.few_shot_cot import select_shots
        from prompts.few_shot_cot import format_examples
        examples = format_examples(select_shots())
        return lambda sample: _few_shot_cot_one(sample, model, params, examples)

    if label == "retrieve_then_classify":
        from runner.retrieve_then_classify import RetrieveThenClassifyPipeline
        pipeline = RetrieveThenClassifyPipeline(model, params)
        return lambda sample: pipeline.run_sample(sample)

    if label == "h_question_pos":
        from runner.h_question import HQuestionPipeline
        pipeline = HQuestionPipeline(model, params, seeder_name="pos", aggregation="aggregated", few_shot=True)
        return lambda sample: pipeline.run_sample(sample)

    if label == "h_question_srl":
        from runner.h_question import HQuestionPipeline
        pipeline = HQuestionPipeline(model, params, seeder_name="srl", aggregation="aggregated", few_shot=True)
        return lambda sample: pipeline.run_sample(sample)

    if label == "bridge_question":
        from runner.bridge_question import BridgeQuestionPipeline
        pipeline = BridgeQuestionPipeline(model, params, aggregation="aggregated", few_shot=True)
        return lambda sample: pipeline.run_sample(sample)

    if label == "p_question_decomposition":
        from runner.p_question import PQuestionPipeline
        pipeline = PQuestionPipeline(model, params, generation="decomposition", aggregation="aggregated", few_shot=True)
        return lambda sample: pipeline.run_sample(sample)

    if label == "p_question_seeded_pos":
        from runner.p_question import PQuestionPipeline
        pipeline = PQuestionPipeline(model, params, generation="seeded", seeder_name="pos", aggregation="aggregated", few_shot=True)
        return lambda sample: pipeline.run_sample(sample)

    if label == "p_question_seeded_srl":
        from runner.p_question import PQuestionPipeline
        pipeline = PQuestionPipeline(model, params, generation="seeded", seeder_name="srl", aggregation="aggregated", few_shot=True)
        return lambda sample: pipeline.run_sample(sample)

    raise ValueError(f"Unknown method label '{label}'")


METHOD_LABELS = [
    "zero_shot",
    "few_shot_cot",
    "retrieve_then_classify",
    "h_question_pos",
    "h_question_srl",
    "bridge_question",
    "p_question_decomposition",
    "p_question_seeded_pos",
    "p_question_seeded_srl",
]


def run_method_concurrent(label: str, samples: list[dict], model: str, params: dict, max_workers: int) -> list[dict]:
    worker = _build_worker(label, model, params)
    results = []
    done = 0

    def _safe_call(sample):
        try:
            return worker(sample)
        except Exception as exc:
            logger.error(f"  {label} | id={sample.get('id')} failed: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sample, result in zip(samples, ex.map(_safe_call, samples)):
            done += 1
            if result is not None:
                results.append(result)
            pred = result.get("prediction") if result else None
            gold = sample.get("label")
            logger.info(f"  [{done}/{len(samples)}] id={sample.get('id')} gold={gold} pred={pred}")

    return results


def _already_done(label: str, model: str) -> bool:
    """True if a results file for this method+model already exists —
    lets a run resume on a fresh machine (e.g. a stronger GPU pod) without
    redoing methods that were already completed and pushed elsewhere."""
    import glob
    safe_model = model.replace("/", "-").replace(":", "-")
    pattern = os.path.join(PRODUCTION_RESULTS_DIR, f"production_train_{label}_{safe_model}*.json")
    return len(glob.glob(pattern)) > 0


def run_all(model: str, max_workers: int, only: list[str] | None = None, limit: int | None = None):
    samples = get_production_samples(N)
    if limit:
        samples = samples[:limit]
    logger.info(f"Loaded {len(samples)} pinned train-split samples for production run (model={model}, workers={max_workers})")

    labels = only if only else METHOD_LABELS
    for label in labels:
        if not limit and _already_done(label, model):
            logger.info(f"SKIP {label} (model={model}) — results file already exists in {PRODUCTION_RESULTS_DIR}")
            continue

        logger.info("#" * 70)
        logger.info(f"PRODUCTION RUN | model={model} | method={label} | n={len(samples)} | workers={max_workers}")
        logger.info("#" * 70)

        try:
            results = run_method_concurrent(label, samples, model, PARAMS, max_workers)
        except Exception as exc:
            logger.error(f"Method {label} failed entirely: {exc}")
            continue

        def _gold(r):
            return r.get("gold_label", r.get("label"))

        correct = sum(1 for r in results if _gold(r) == r.get("prediction"))
        n_results = len(results)
        acc = correct / n_results if n_results else 0.0
        logger.info(f"  {label}: {correct}/{n_results} = {acc:.4f}")

        os.makedirs(PRODUCTION_RESULTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"production_train_{label}" if not limit else f"pilot{limit}_{label}"
        safe_model = model.replace("/", "-").replace(":", "-")
        path = os.path.join(PRODUCTION_RESULTS_DIR, f"{tag}_{safe_model}_{ts}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved {len(results)} results for {label} -> {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python run_production_train.py <model> [max_workers={DEFAULT_MAX_WORKERS}] [--only label1,label2] [--limit N]")
        sys.exit(1)

    args = sys.argv[1:]
    only_labels = None
    limit_n = None
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--only":
            only_labels = args[i + 1].split(",")
            i += 2
        elif args[i] == "--limit":
            limit_n = int(args[i + 1])
            i += 2
        else:
            positional.append(args[i])
            i += 1

    model = positional[0]
    max_workers = int(positional[1]) if len(positional) > 1 else DEFAULT_MAX_WORKERS
    setup_logger("production_train", model)
    run_all(model, max_workers, only=only_labels, limit=limit_n)
