import os
import logging
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

load_dotenv()

LAB_API_KEY = os.getenv("LAB_API_KEY", "")
LAB_API_URL = os.getenv("LAB_API_URL", "http://100.110.96.82:8000/chat")

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_API_URL = "https://router.huggingface.co/featherless-ai/v1"

# HuggingFace model ID for each friendly name
HF_MODEL_MAP = {
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1"

MODELS = {
    # ── Lab (Tailscale) ───────────────────────────────────────────────────────
    "llama3.1-8b":       "open_source",
    "llama3.1-70b":      "open_source",
    "qwen2.5-32b":       "open_source",
    "qwen2.5-72b":       "open_source",
    "qwen3.6-35b":       "open_source",
    "qwen3.6-35b-ablit": "open_source",
    "gpt-oss-20b":       "open_source",
    "gemma4-31b":        "open_source",
    "nemotron-70b":      "open_source",
    # ── Hugging Face ──────────────────────────────────────────────────────────
    "qwen2.5-32b-instruct": "huggingface",
    # ── Groq ──────────────────────────────────────────────────────────────────
    "llama-3.1-8b-instant":    "groq",
    "llama-3.3-70b-versatile": "groq",
    "qwen/qwen3-32b":          "groq",
}

DEFAULT_PARAMS = {
    "temperature": 0.0,
    "max_tokens":  2048,
}

RESULTS_DIR = "results"
LOGS_DIR    = "logs"
DATA_PATH   = "data/ConTRoL-dataset"


class NLIOutput(BaseModel):
    label:       Literal["Entailment", "Contradiction", "Neutral"]
    explanation: str


class QuestionListOutput(BaseModel):
    questions: list[str]


class PDTBQuestion(BaseModel):
    id:     str
    probes: str
    q:      str


class PDTBOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hypothesis:      str
    connective:      str
    connective_type: Literal["explicit", "implicit"] = Field(alias="connectivetype")
    pdtb_sense:      str = Field(alias="pdtbsense")
    arg1:            str
    arg2:            str
    questions:       list[PDTBQuestion]


class QAPair(BaseModel):
    question: str
    answer:   str


class HDQDOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    qa_pairs:    list[QAPair] = Field(alias="qapairs")
    comparisons: list[str]
    label:       Literal["Entailment", "Contradiction", "Neutral"]
    explanation: str


# ── Q2 Pipeline schemas ───────────────────────────────────────────────────────

# ── Q1 output ────────────────────────────────────────────────────────────────
class Q2QuestionOutput(BaseModel):
    anchors:   list[str]
    questions: list[str]


# ── Q2 audit output ───────────────────────────────────────────────────────────
class AuditTableRow(BaseModel):
    question:                       str
    target_anchor:                  str
    verbatim_premise_evidence_list: list[str]
    integrated_premise_tags:        str
    found:                          bool


class Q2AuditOutput(BaseModel):
    audit_table_decomposition: list[AuditTableRow]
    matrix_cross_check_flags:  list[str]
    label:                     Literal["Entailment", "Contradiction", "Neutral"]
    explanation:               str


# ── P-Question Pipeline constants ─────────────────────────────────────────────
# Swap these two lines to change ablation variant — no other code changes needed.
P_QUESTION_TOP_K:           int = 3
P_QUESTION_ALIGNMENT_METRIC: str = "ROUGE_L"   # "ROUGE_L" | "BLEU"
P_QUESTION_STAGE2_MODE:      str = "concatenated"  # "concatenated" | "majority_vote"


# ── P-Question Pipeline schemas ───────────────────────────────────────────────

class PQuestionListOutput(BaseModel):
    """Stage 1a output: factual questions the premise directly answers."""
    questions: list[str]

    @field_validator("questions", mode="before")
    @classmethod
    def coerce_questions_to_list(cls, v):
        # Some models return a newline-separated string instead of a JSON array
        if isinstance(v, str):
            return [line.strip() for line in v.splitlines() if line.strip()]
        return v

    @field_validator("questions", mode="after")
    @classmethod
    def drop_empty_questions(cls, v):
        return [q.strip() for q in v if q.strip()]


class PEvidenceGatheringOutput(BaseModel):
    """Stage 1c Step 1: all premise sentences relevant to a question."""
    evidence_sentences: list[str]  # every relevant span, verbatim or close paraphrase
    has_evidence:       bool       # False only when no relevant sentence exists at all

    @field_validator("evidence_sentences", mode="before")
    @classmethod
    def coerce_sentences_to_list(cls, v):
        # Guard against model returning a single string instead of a JSON array
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        return v

    @field_validator("evidence_sentences", mode="after")
    @classmethod
    def drop_empty_sentences(cls, v):
        return [s.strip() for s in v if s.strip()]

    @field_validator("has_evidence", mode="before")
    @classmethod
    def coerce_has_evidence_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        return v

    @model_validator(mode="after")
    def sync_has_evidence_with_sentences(self) -> "PEvidenceGatheringOutput":
        # If the model returned sentences but forgot to set has_evidence=True, fix it
        if self.evidence_sentences and not self.has_evidence:
            self.has_evidence = True
        # If the model set has_evidence=True but returned an empty list, correct it
        if not self.evidence_sentences and self.has_evidence:
            self.has_evidence = False
        return self


class PAnswerOutput(BaseModel):
    """Stage 1c Step 2: one-sentence answer synthesised from gathered evidence."""
    answer:       str   # answer text or exactly "[UNANSWERABLE]"
    unanswerable: bool

    @field_validator("answer", mode="before")
    @classmethod
    def strip_answer(cls, v):
        return str(v).strip() if v is not None else "[UNANSWERABLE]"

    @field_validator("unanswerable", mode="before")
    @classmethod
    def coerce_unanswerable_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        return v

    @model_validator(mode="after")
    def sync_unanswerable_with_answer(self) -> "PAnswerOutput":
        # If the answer text signals unanswerable but the flag wasn't set, fix it
        if "[UNANSWERABLE]" in self.answer.upper() and not self.unanswerable:
            self.unanswerable = True
        return self


class PNLIOutput(BaseModel):
    """Stage 2 output: NLI label for evidence → hypothesis."""
    label: Literal["Entailment", "Contradiction", "Neutral"]

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label_case(cls, v):
        # Handle ENTAILMENT, neutral, CONTRADICTION, etc. from any model
        if isinstance(v, str):
            title = v.strip().title()
            if title in {"Entailment", "Contradiction", "Neutral"}:
                return title
        return v  # pass through unchanged; Pydantic raises a clear error

def setup_logger(experiment: str, model: str) -> logging.Logger:
    from datetime import datetime
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "-")
    log_file = os.path.join(LOGS_DIR, f"{experiment}_{safe_model}_{timestamp}.log")

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    handler_file    = logging.FileHandler(log_file, encoding="utf-8")
    handler_console = logging.StreamHandler()
    for h in (handler_file, handler_console):
        h.setFormatter(fmt)

    log = logging.getLogger("control")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(handler_file)
    log.addHandler(handler_console)
    return log


logger = logging.getLogger("control")


def get_structured_llm(model: str, schema, params: dict = DEFAULT_PARAMS):
    llm = get_llm(model, params)
    if MODELS.get(model) == "groq":
        return llm.with_structured_output(schema, method="json_mode")
    return llm.with_structured_output(schema)


def get_llm(model: str, params: dict = DEFAULT_PARAMS):
    if MODELS.get(model) == "huggingface":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=HF_MODEL_MAP[model],
            base_url=HF_API_URL,
            api_key=HF_API_KEY,
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 2048),
        )
    if MODELS.get(model) == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            base_url=GROQ_API_URL,
            api_key=GROQ_API_KEY,
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 2048),
        )
    from open_source_llm import OpenSourceChatModel
    return OpenSourceChatModel(
        model=model,
        api_url=LAB_API_URL,
        api_key=LAB_API_KEY,
        temperature=params.get("temperature", 0.0),
        max_tokens=params.get("max_tokens", 512),
    )
