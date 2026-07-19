from langchain_core.messages import SystemMessage, HumanMessage
from config.config import DEFAULT_PARAMS, get_structured_llm, NLIOutput, logger
from utils.retry import call_with_retry
from prompts.zero_shot import SYSTEM_PROMPT, USER_PROMPT


def call(system: str, user: str, model: str, params: dict) -> NLIOutput:
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    return call_with_retry(get_structured_llm(model, NLIOutput, params).invoke, messages)


def run(samples: list[dict], model: str, params: dict = DEFAULT_PARAMS) -> list[dict]:
    logger.info("=" * 60)
    logger.info("Experiment : zero_shot")
    logger.info(f"Model      : {model}")
    logger.info(f"Temperature: {params.get('temperature')}")
    logger.info(f"Max tokens : {params.get('max_tokens')}")
    logger.info(f"Samples    : {len(samples)}")
    logger.info("=" * 60)

    results = []

    for i, sample in enumerate(samples):
        system = SYSTEM_PROMPT
        user = USER_PROMPT.format(premise=sample["premise"], hypothesis=sample["hypothesis"])
        try:
            output = call(system, user, model, params)
        except Exception as exc:
            logger.warning(f"[{i+1}/{len(samples)}] id={sample.get('id')} | structured output parsing failed, skipping: {exc}")
            continue

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
