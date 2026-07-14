from __future__ import annotations
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from config.config import DEFAULT_PARAMS, HDQDOutput, QuestionListOutput, get_structured_llm, logger
from experiments.archived_methods.prompts.hdqd_pipeline import (
    HDQD_QUESTION_SYSTEM_PROMPT,
    HDQD_QUESTION_USER_PROMPT,
    HDQD_ANSWER_SYSTEM_PROMPT,
    HDQD_ANSWER_USER_PROMPT,
)


def call_questions(hypothesis: str, model: str, params: dict) -> QuestionListOutput | None:
    messages = [
        SystemMessage(content=HDQD_QUESTION_SYSTEM_PROMPT),
        HumanMessage(content=HDQD_QUESTION_USER_PROMPT.format(hypothesis=hypothesis)),
    ]
    return get_structured_llm(model, QuestionListOutput, params).invoke(messages)


def call_answer(premise: str, hypothesis: str, questions: list[str], model: str, params: dict) -> HDQDOutput | None:
    formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    messages = [
        SystemMessage(content=HDQD_ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=HDQD_ANSWER_USER_PROMPT.format(
            premise=premise,
            hypothesis=hypothesis,
            questions=formatted_questions,
        )),
    ]
    return get_structured_llm(model, HDQDOutput, params).invoke(messages)


def run(samples: list[dict], model: str, params: dict = DEFAULT_PARAMS) -> list[dict]:
    logger.info("=" * 60)
    logger.info("Experiment : hdqd_pipeline")
    logger.info(f"Model      : {model}")
    logger.info(f"Temperature: {params.get('temperature')}")
    logger.info(f"Max tokens : {params.get('max_tokens')}")
    logger.info(f"Samples    : {len(samples)}")
    logger.info("=" * 60)

    results = []

    for i, sample in enumerate(samples):
        sample_id = sample.get("id")

        # Step 1: decompose hypothesis into targeted questions
        q_output = call_questions(sample["hypothesis"], model, params)
        if q_output is None:
            logger.warning(f"[{i+1}/{len(samples)}] id={sample_id} | question generation failed, skipping")
            continue

        questions = q_output.questions
        logger.debug(f"[{i+1}/{len(samples)}] id={sample_id} | questions={questions}")

        # Step 2: answer questions against the premise and predict NLI label
        a_output = call_answer(sample["premise"], sample["hypothesis"], questions, model, params)
        if a_output is None:
            logger.warning(f"[{i+1}/{len(samples)}] id={sample_id} | answer/prediction failed, skipping")
            continue

        results.append({
            "id": sample_id,
            "premise": sample["premise"],
            "hypothesis": sample["hypothesis"],
            "label": sample["label"],
            "prediction": a_output.label,
            "explanation": a_output.explanation,
            "qa_pairs": [{"question": qa.question, "answer": qa.answer} for qa in a_output.qa_pairs],
            "comparisons": a_output.comparisons,
        })

        logger.info(f"[{i+1}/{len(samples)}] id={sample_id} | gold={sample['label']} | pred={a_output.label}")

    logger.info("=" * 60)
    logger.info(f"Done | {len(results)} samples processed")
    logger.info("=" * 60)
    return results
