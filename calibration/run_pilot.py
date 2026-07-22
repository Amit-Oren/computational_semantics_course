"""
Calibration pilot — Task 1 (timing/token harness) + Task 2 (scale decisions).

Runs all 9 project methods against a fixed, pinned sample set from the
ConTRoL train split, measuring wall-clock time, token usage, and LLM call
count per sample. Projects total time for the full train split (n=6719)
and, per Task 2's rule, decides whether each method/model combination runs
at full scale, a reduced subsample, or a small directional-only pass.

Does NOT start any full-scale or subsampled production run (Task 3) — this
script stops after printing the decision table, per the project's explicit
gate: full runs require human confirmation of MAX_PROJECTED_HOURS first.

Token-usage capture is instrumented here only, via a monkeypatch of
config.config._StructuredOutput.invoke scoped to this process — it does
NOT modify the shared production code path other runs depend on. It
duplicates a few lines of that method's post-processing (fence-stripping +
the JSON-repair regex) but references the live module-level regex objects
directly, so it can't silently drift out of sync if those are changed later.
"""
from __future__ import annotations

import importlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.config as cfg
from config.config import setup_logger, logger
from data.data import load_split

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
PINNED_IDS_PATH = os.path.join(CALIB_DIR, "pinned_ids.json")

N_PINNED = 20
N_LIGHT = 10  # subset of the same pinned IDs used for the two lighter-pass models
TRAIN_N = 4000  # project's chosen full-scale target (not the full 6719-sample train split)
MAX_PROJECTED_HOURS = 5  # confirm before Task 3 — see module docstring

PARAMS = {"temperature": 0.0, "max_tokens": 4096}

# (label, runner_module_name, kwargs passed to that module's run())
# Ordered cheap -> heavy, so cheap/fast methods surface problems early.
METHOD_CONFIGS = [
    ("zero_shot",                "zero_shot",              {}),
    ("few_shot_cot",              "few_shot_cot",            {}),
    ("retrieve_then_classify",    "retrieve_then_classify",  {}),
    ("h_question_pos",            "h_question",              {"seeder_name": "pos", "aggregation": "aggregated", "few_shot": True}),
    ("h_question_srl",            "h_question",              {"seeder_name": "srl", "aggregation": "aggregated", "few_shot": True}),
    ("bridge_question",           "bridge_question",         {"aggregation": "aggregated", "few_shot": True}),
    ("p_question_decomposition",  "p_question",              {"generation": "decomposition", "aggregation": "aggregated", "few_shot": True}),
    ("p_question_seeded_pos",     "p_question",              {"generation": "seeded", "seeder_name": "pos", "aggregation": "aggregated", "few_shot": True}),
    ("p_question_seeded_srl",     "p_question",              {"generation": "seeded", "seeder_name": "srl", "aggregation": "aggregated", "few_shot": True}),
]


# ─── Pinned sample selection ───────────────────────────────────────────────

def get_pinned_ids() -> list[str]:
    if os.path.exists(PINNED_IDS_PATH):
        with open(PINNED_IDS_PATH) as f:
            return json.load(f)
    train = load_split("train")
    rng = random.Random(42)
    picked = rng.sample(train, N_PINNED)
    ids = [s["id"] for s in picked]
    with open(PINNED_IDS_PATH, "w") as f:
        json.dump(ids, f, indent=2)
    logger.info(f"Generated {len(ids)} new pinned IDs -> {PINNED_IDS_PATH}")
    return ids


def samples_for_ids(ids: list[str]) -> list[dict]:
    train = load_split("train")
    by_id = {s["id"]: s for s in train}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"Pinned IDs not found in train split: {missing}")
    return [by_id[i] for i in ids]


# ─── Calibration-only usage instrumentation (does not touch production code) ───

_usage_log: list[dict] = []


def _extract_usage(response) -> dict:
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        return {
            "prompt_tokens":     usage_metadata.get("input_tokens", 0),
            "completion_tokens": usage_metadata.get("output_tokens", 0),
            "total_tokens":      usage_metadata.get("total_tokens", 0),
        }
    response_metadata = getattr(response, "response_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if token_usage:
        return {
            "prompt_tokens":     token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens":      token_usage.get("total_tokens", 0),
        }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _instrumented_invoke(self, messages):
    response = self._llm.invoke(messages)
    _usage_log.append(_extract_usage(response))
    content = getattr(response, "content", str(response)).strip()
    m = cfg._JSON_FENCE_RE.search(content)
    if m:
        content = m.group(1).strip()
    content = cfg._BARE_JSON_VALUE_RE.sub(r'"\1": "\2"\3', content)
    return self._schema.model_validate_json(content)


def install_usage_instrumentation():
    cfg._StructuredOutput.invoke = _instrumented_invoke
    logger.info("Installed calibration-only usage instrumentation on _StructuredOutput.invoke")


# ─── Per-sample, per-method measurement ────────────────────────────────────

def run_one_sample(method_module, sample: dict, model: str, kwargs: dict) -> dict:
    _usage_log.clear()
    t0 = time.time()
    try:
        results = method_module.run([sample], model=model, params=PARAMS, **kwargs)
        ok = bool(results)
    except Exception as exc:
        logger.error(f"  sample {sample['id']} failed: {exc}")
        ok = False
    elapsed = time.time() - t0

    calls = len(_usage_log)
    total_tokens = sum(u["total_tokens"] for u in _usage_log)
    return {
        "sample_id":    sample["id"],
        "ok":           ok,
        "seconds":      elapsed,
        "llm_calls":    calls,
        "total_tokens": total_tokens,
    }


def run_pilot_for_model(model: str, ids: list[str], save_path: str | None = None) -> dict:
    samples = samples_for_ids(ids)
    per_method = {}

    for label, module_name, kwargs in METHOD_CONFIGS:
        logger.info("=" * 70)
        logger.info(f"CALIBRATION | model={model} | method={label} | n={len(samples)}")
        logger.info("=" * 70)
        module = importlib.import_module(f"runner.{module_name}")

        per_sample = []
        for i, sample in enumerate(samples):
            row = run_one_sample(module, sample, model, kwargs)
            per_sample.append(row)
            logger.info(
                f"  [{i+1}/{len(samples)}] id={row['sample_id']} "
                f"ok={row['ok']} sec={row['seconds']:.2f} "
                f"calls={row['llm_calls']} tokens={row['total_tokens']}"
            )

        ok_rows = [r for r in per_sample if r["ok"]]
        n_ok = len(ok_rows) or 1  # avoid div-by-zero if every sample failed
        mean_sec    = sum(r["seconds"] for r in ok_rows) / n_ok
        mean_tokens = sum(r["total_tokens"] for r in ok_rows) / n_ok
        mean_calls  = sum(r["llm_calls"] for r in ok_rows) / n_ok
        projected_hours = (mean_sec * TRAIN_N) / 3600.0

        per_method[label] = {
            "model":              model,
            "n_run":              len(samples),
            "n_ok":               len(ok_rows),
            "mean_sec_per_sample":     mean_sec,
            "mean_tokens_per_sample":  mean_tokens,
            "mean_calls_per_sample":   mean_calls,
            "projected_hours_n6719":  projected_hours,  # key name kept for compat; value uses current TRAIN_N
            "per_sample":         per_sample,
        }
        logger.info(
            f"  SUMMARY {label}: mean_sec={mean_sec:.2f} mean_tokens={mean_tokens:.0f} "
            f"mean_calls={mean_calls:.1f} projected_hours(n={TRAIN_N})={projected_hours:.2f}"
        )

        # Save incrementally after every method, not just at the end — a kill
        # or crash partway through a 9-method run no longer loses everything
        # already completed (this bit us earlier this session).
        if save_path:
            with open(save_path, "w") as f:
                json.dump(per_method, f, indent=2, default=str)

    return per_method


def print_summary_table(model: str, per_method: dict):
    print(f"\n=== Pilot summary: {model} ===")
    rows = sorted(per_method.items(), key=lambda kv: kv[1]["projected_hours_n6719"], reverse=True)
    print(f"{'METHOD':<28} {'SEC/SAMPLE':>10} {'TOKENS/SMPL':>12} {'CALLS/SMPL':>11} {'PROJ.HRS(n=6719)':>17}")
    for label, m in rows:
        print(
            f"{label:<28} {m['mean_sec_per_sample']:>10.2f} {m['mean_tokens_per_sample']:>12.0f} "
            f"{m['mean_calls_per_sample']:>11.1f} {m['projected_hours_n6719']:>17.2f}"
        )


# ─── Task 2: scale decisions ────────────────────────────────────────────────

def decide_scale(projected_hours: float) -> dict:
    if projected_hours <= MAX_PROJECTED_HOURS:
        return {"scale": "full", "n": TRAIN_N}
    if projected_hours <= 3 * MAX_PROJECTED_HOURS:
        reduced_n = int(TRAIN_N * MAX_PROJECTED_HOURS / projected_hours)
        return {"scale": "subsampled", "n": reduced_n}
    return {
        "scale": "directional",
        "n": 500,
        "note": "directional only — exceeds 3x ceiling, treat as pilot-scale per project convention",
    }


def build_decision_table(all_results: dict) -> dict:
    """all_results: {model: {method_label: summary_dict}}"""
    decisions = {}
    for model, per_method in all_results.items():
        for label, summary in per_method.items():
            decision = decide_scale(summary["projected_hours_n6719"])
            decisions[f"{label}|{model}"] = {
                "method": label,
                "model": model,
                "projected_hours": summary["projected_hours_n6719"],
                **decision,
            }
    return decisions


def save_subsample_ids(decisions: dict, pinned_ids: list[str]):
    """For methods marked subsampled/directional, draw ONE fixed ID subset
    (from the full train split, not just the pinned pilot set) shared across
    all models for that method, so McNemar comparisons stay valid."""
    train = load_split("train")
    by_method: dict[str, dict] = {}
    for key, d in decisions.items():
        if d["scale"] in ("subsampled", "directional"):
            by_method.setdefault(d["method"], d["n"])
            by_method[d["method"]] = max(by_method[d["method"]], d["n"])

    for method, n in by_method.items():
        path = os.path.join(CALIB_DIR, f"subsample_ids_{method}.json")
        if os.path.exists(path):
            continue
        rng = random.Random(hash(method) & 0xFFFFFFFF)
        picked = rng.sample(train, min(n, len(train)))
        with open(path, "w") as f:
            json.dump([s["id"] for s in picked], f, indent=2)
        logger.info(f"Saved {len(picked)} subsample IDs for '{method}' -> {path}")


def print_decision_table(decisions: dict):
    print("\n=== Scale decision table ===")
    print(f"{'METHOD':<28} {'MODEL':<14} {'PROJ.HRS':>9} {'SCALE':<13} {'N':>6}")
    for d in sorted(decisions.values(), key=lambda d: (d["method"], d["model"])):
        print(f"{d['method']:<28} {d['model']:<14} {d['projected_hours']:>9.2f} {d['scale']:<13} {d['n']:>6}")


# ─── Main ───────────────────────────────────────────────────────────────────

PRIMARY_MODEL = "qwen2.5-32b"
LIGHT_MODELS  = ("gpt-oss-20b", "gemma4-31b")


def run_and_save(model: str) -> dict:
    pinned_ids = get_pinned_ids()
    ids = pinned_ids if model == PRIMARY_MODEL else pinned_ids[:N_LIGHT]
    out_path = os.path.join(CALIB_DIR, f"{model}_pilot_results.json")
    results = run_pilot_for_model(model, ids, save_path=out_path)
    print_summary_table(model, results)
    print(f"\nSaved -> {out_path}")
    return results


def run_decision_table():
    """Loads whatever *_pilot_results.json files already exist and builds
    the Task 2 scale-decision table. Run this once all three models'
    pilots have completed."""
    all_results = {}
    for model in (PRIMARY_MODEL,) + LIGHT_MODELS:
        path = os.path.join(CALIB_DIR, f"{model}_pilot_results.json")
        if not os.path.exists(path):
            print(f"Missing {path} — run `python run_pilot.py {model}` first.")
            return
        with open(path) as f:
            all_results[model] = json.load(f)

    pinned_ids = get_pinned_ids()
    decisions = build_decision_table(all_results)
    with open(os.path.join(CALIB_DIR, "scale_decisions.json"), "w") as f:
        json.dump(decisions, f, indent=2)
    save_subsample_ids(decisions, pinned_ids)
    print_decision_table(decisions)
    print(
        f"\nTask 3 gate: no full-scale or subsampled production run has been started. "
        f"MAX_PROJECTED_HOURS={MAX_PROJECTED_HOURS} — confirm this ceiling and the per-method "
        f"scale decisions above before proceeding."
    )


if __name__ == "__main__":
    setup_logger("calibration_pilot", "multi")
    install_usage_instrumentation()

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            f"  python run_pilot.py {PRIMARY_MODEL}      # n={N_PINNED} pilot on the primary model\n"
            f"  python run_pilot.py <light-model>        # n={N_LIGHT} pilot, one of: {LIGHT_MODELS}\n"
            "  python run_pilot.py --decide              # build Task 2 scale-decision table "
            "(run after all three models' pilots are done)"
        )
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "--decide":
        run_decision_table()
    elif arg == PRIMARY_MODEL or arg in LIGHT_MODELS:
        run_and_save(arg)
    else:
        print(f"Unknown model '{arg}'. Expected one of: {PRIMARY_MODEL}, {', '.join(LIGHT_MODELS)}, or --decide")
        sys.exit(1)
