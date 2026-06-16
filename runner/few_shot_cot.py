from langchain_core.messages import SystemMessage, HumanMessage
from config.config import DEFAULT_PARAMS, get_llm, NLIOutput, logger
from data.data import load_split
from prompts.few_shot_cot import SYSTEM_PROMPT, USER_PROMPT, format_examples

LABEL_ORDER = ["Entailment", "Contradiction", "Neutral"]

# fixed few-shot demonstrations
# manually verified as correctly-labelled item per label (Entailment, Contradiction, Neutral), each from a
# different premise. 
SHOT_IDS = ["id_193", "id_123", "id_315"]


def select_shots() -> list[dict]:
    """The three fixed few-shot demonstrations, by uid from the dev split.

    Same 3 shots in every prompt and across runs.
    """
    by_id = {s["id"]: s for s in load_split("dev")}
    shots = [by_id[uid] for uid in SHOT_IDS]
    shots.sort(key=lambda s: LABEL_ORDER.index(s["label"]))
    return shots


def call(system: str, user: str, model: str, params: dict) -> NLIOutput:
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    return get_llm(model, params).with_structured_output(NLIOutput).invoke(messages)


def run(samples: list[dict], model: str, params: dict = DEFAULT_PARAMS) -> list[dict]:
    shots = select_shots()
    examples = format_examples(shots)

    logger.info("=" * 60)
    logger.info("Experiment : few_shot_cot (3-shot, one per label)")
    logger.info(f"Model      : {model}")
    logger.info(f"Temperature: {params.get('temperature')}")
    logger.info(f"Max tokens : {params.get('max_tokens')}")
    logger.info(f"Shots      : {len(shots)} ({', '.join(s['label'] for s in shots)})")
    logger.info(f"Samples    : {len(samples)}")
    logger.info("=" * 60)

    results = []

    for i, sample in enumerate(samples):
        user = USER_PROMPT.format(
            examples=examples,
            premise=sample["premise"],
            hypothesis=sample["hypothesis"],
        )
        output = call(SYSTEM_PROMPT, user, model, params)

        if output is None:
            logger.warning(f"[{i+1}/{len(samples)}] id={sample.get('id')} | structured output parsing failed, skipping")
            continue

        results.append({
            "id": sample.get("id"),
            "label": sample["label"],
            "prediction": output.label,
            "explanation": output.explanation,
        })

        logger.info(f"[{i+1}/{len(samples)}] id={sample.get('id')} | gold={sample['label']} | pred={output.label}")

    logger.info("=" * 60)
    logger.info(f"Done | {len(results)} samples processed")
    logger.info("=" * 60)
    return results
