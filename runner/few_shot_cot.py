from langchain_core.messages import SystemMessage, HumanMessage
from config.config import DEFAULT_PARAMS, get_llm, NLIOutput, logger
from prompts.few_shot_cot import SYSTEM_PROMPT, USER_PROMPT


def call(system: str, user: str, model: str, params: dict) -> NLIOutput:
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    return get_llm(model, params).with_structured_output(NLIOutput).invoke(messages)


def run(samples: list[dict], model: str, params: dict = DEFAULT_PARAMS) -> list[dict]:
    logger.info("few_shot_cot not implemented yet")
    return []
