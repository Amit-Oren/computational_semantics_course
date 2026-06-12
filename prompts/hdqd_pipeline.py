def build_construction_prompt(hypothesis: str) -> str:
    """Stage 1: hypothesis → sub-questions targeting needed evidence."""
    return f"""You are analyzing a hypothesis that needs to be verified against a passage.

Hypothesis:
{hypothesis}

Generate 4-5 focused sub-questions that identify the specific evidence needed from the passage to determine whether this hypothesis is Entailed, Contradicted, or Neutral.
Consider these reasoning types: verbal logical, information integration, temporal/mathematical, coreferential, and analytical.

Return only a numbered list of questions."""


def build_integration_prompt(premise: str, hypothesis: str, sub_questions: list[str]) -> str:
    """Stage 2: passage + sub-questions → verdict."""
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))
    return f"""You are given a passage, a hypothesis, and a set of sub-questions.

Passage:
{premise}

Hypothesis:
{hypothesis}

Sub-questions:
{questions_text}

Answer each sub-question using only evidence from the passage, then give a final verdict.

Return your answers as a list (one per sub-question), a final label (Entailment, Contradiction, or Neutral), and a one-sentence justification."""
