"""Few-shot prompt: three demonstration examples followed by the item to classify.
The system prompt is the one used for zero-shot (imported below); the only addition
the model sees is the three examples. Standard 3-shot prompting, not chain-of-thought."""
from prompts.zero_shot import SYSTEM_PROMPT

EXAMPLE_TEMPLATE = """Example {n}

PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}

ANSWER:
{{"label": "{label}", "explanation": "{explanation}"}}"""

SHOT_EXPLANATION = "The label follows from the relationship between the premise and the hypothesis."

USER_PROMPT = """Here are three solved examples:

{examples}

Now solve the next item.

PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}"""


def format_examples(shots: list[dict]) -> str:
    blocks = [
        EXAMPLE_TEMPLATE.format(
            n=i,
            premise=s["premise"],
            hypothesis=s["hypothesis"],
            label=s["label"],
            explanation=s.get("explanation", SHOT_EXPLANATION),
        )
        for i, s in enumerate(shots, start=1)
    ]
    return "\n\n".join(blocks)
