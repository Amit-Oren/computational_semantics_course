import os
import logging
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

load_dotenv(override=True)

LAB_API_KEY = os.getenv("LAB_LLM_TOKEN", "")
LAB_API_URL = "http://lab-server.tailbdc662.ts.net:8000/v1"

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_API_URL = "https://router.huggingface.co/featherless-ai/v1"

# HuggingFace model ID for each friendly name
HF_MODEL_MAP = {
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "llama-3-8b-instruct":  "meta-llama/Meta-Llama-3-8B-Instruct",
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
    "llama-3-8b-instruct":  "huggingface",
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

class Q2QuestionOutput(BaseModel):
    """Stage 1 output: hypothesis anchors + verification questions."""
    anchors:   list[str]
    questions: list[str]


# ── P-Question Pipeline constants ─────────────────────────────────────────────
# Stage 1b's question-selection is swappable behind two orthogonal flags —
# see utils/question_selectors.py. Only Stage 1b changes across ablation
# runs; Stage 1a generation, Stage 1c locate_and_answer, and Stage 2
# classify_evidence are identical regardless of which combo is active.
P_QUESTION_TOP_K:     int   = 3
P_QUESTION_SELECTOR:  str   = "rouge_l"   # "rouge_l" | "embedding" | "llm_relevance"
P_QUESTION_SELECTION: str   = "topk"      # "topk" | "mmr"
P_QUESTION_MMR_LAMBDA: float = 0.7        # 1.0 reduces MMR exactly to top-K

# Stage 1a decomposes the premise into atomic facts + relations (one question
# each) — this can produce 20-40 candidates for a ~500-word premise, well
# above the old free-form cap. Capped here to bound Stage 1b/1c cost;
# relation-type questions (scarce, high-value) are kept first, fact-type
# fills the remainder. See runner/p_question.py._cap_questions.
P_QUESTION_MAX_QUESTIONS: int = 30


# ── P-Question Pipeline schemas ───────────────────────────────────────────────

class DecomposedQuestion(BaseModel):
    """One Stage 1a unit: an atomic fact or a relation/comparison, already
    phrased as a question."""
    q:    str
    type: Literal["fact", "relation"] = "fact"

    @field_validator("q", mode="before")
    @classmethod
    def coerce_q_str(cls, v):
        return str(v).strip() if v is not None else ""

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v):
        if isinstance(v, str) and v.strip().lower() == "relation":
            return "relation"
        return "fact"


class PQuestionListOutput(BaseModel):
    """Stage 1a output: atomic-fact + relation decomposition of the premise."""
    questions: list[DecomposedQuestion]

    @field_validator("questions", mode="before")
    @classmethod
    def coerce_questions_to_list(cls, v):
        # Some models return a newline-separated string, or a plain list of
        # strings, instead of the {"q", "type"} object list.
        if isinstance(v, str):
            return [{"q": line.strip(), "type": "fact"} for line in v.splitlines() if line.strip()]
        if isinstance(v, list):
            return [
                {"q": item, "type": "fact"} if isinstance(item, str) else item
                for item in v
            ]
        return v

    @field_validator("questions", mode="after")
    @classmethod
    def drop_empty_questions(cls, v):
        return [item for item in v if item.q]


class LLMRelevanceOutput(BaseModel):
    """Stage 1b llm_relevance scorer output: 0-5 relevance-to-verdict rating."""
    score: int

    @field_validator("score", mode="before")
    @classmethod
    def coerce_int(cls, v):
        if isinstance(v, str):
            import re
            m = re.search(r"-?\d+", v)
            return int(m.group()) if m else 0
        return v

    @field_validator("score", mode="after")
    @classmethod
    def clamp_range(cls, v):
        return max(0, min(5, v))


# ── H-Question Pipeline schemas ───────────────────────────────────────────────

class HQuestionsOutput(BaseModel):
    """Stage 1 output: probe questions derived from the hypothesis."""
    questions: list[str]

    @field_validator("questions", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if isinstance(v, str):
            return [line.strip() for line in v.splitlines() if line.strip()]
        return v

    @field_validator("questions", mode="after")
    @classmethod
    def drop_empty(cls, v):
        return [q.strip() for q in v if q.strip()]


# ── H-Multihop Pipeline schemas ───────────────────────────────────────────────

class HMDecompOutput(BaseModel):
    """Stage 1: ordered list of 2-3 atomic sub-questions."""
    sub_questions: list[str]

    @field_validator("sub_questions", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if isinstance(v, str):
            return [line.strip() for line in v.splitlines() if line.strip()]
        return v

    @field_validator("sub_questions", mode="after")
    @classmethod
    def drop_empty_and_cap(cls, v):
        cleaned = [q.strip() for q in v if q.strip()]
        return cleaned[:3]


class HMAnswerOutput(BaseModel):
    """Per-hop output: answer text and answerable flag (h_multihop's own
    context-chained hop answerer — kept separate from the shared AnswerOutput
    used by locate_and_answer, since hops take a running context argument)."""
    answer:     str
    answerable: bool

    @field_validator("answer", mode="before")
    @classmethod
    def coerce_str(cls, v):
        return str(v).strip() if v is not None else "NOT_ANSWERABLE"

    @field_validator("answerable", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        return v

    @model_validator(mode="after")
    def sync_answerable(self) -> "HMAnswerOutput":
        if "NOT_ANSWERABLE" in self.answer.upper() and self.answerable:
            self.answerable = False
        if "NOT_ANSWERABLE" not in self.answer.upper() and not self.answerable:
            self.answerable = True
        return self


# ── Shared answering + classification schemas ─────────────────────────────────
# Used by utils/locator_extractor.py and prompts/shared_classifier.py, the two
# components shared across q2_pipeline, p_question, h_question, and h_multihop.

class LocateOutput(BaseModel):
    """Locator output: sentence indices from the numbered premise."""
    indices: list[int]

    @field_validator("indices", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if isinstance(v, str):
            import re
            return [int(n) for n in re.findall(r"\d+", v)]
        return v


class AnswerOutput(BaseModel):
    """Answer Extractor output: answer text and answerable flag."""
    answer:     str
    answerable: bool

    @field_validator("answer", mode="before")
    @classmethod
    def coerce_str(cls, v):
        return str(v).strip() if v is not None else "NOT_ANSWERABLE"

    @field_validator("answerable", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        return v

    @model_validator(mode="after")
    def sync_answerable(self) -> "AnswerOutput":
        if "NOT_ANSWERABLE" in self.answer.upper() and self.answerable:
            self.answerable = False
        if "NOT_ANSWERABLE" not in self.answer.upper() and not self.answerable:
            self.answerable = True
        return self


class ClassifyOutput(BaseModel):
    """Shared classifier output: one NLI label over a full evidence block."""
    label: Literal["Entailment", "Contradiction", "Neutral"]

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label(cls, v):
        if isinstance(v, str):
            title = v.strip().title()
            if title in {"Entailment", "Contradiction", "Neutral"}:
                return title
        return v


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


# A request that never gets a response (dead connection, server hang) would
# otherwise block forever — langchain_openai/ChatOpenAI has no default
# timeout. Capped here so a hung call raises instead of stalling a run
# indefinitely.
#
# max_retries=0 is just as important: the openai SDK's own default
# (max_retries=2, i.e. 3 attempts) retries silently *inside* a single
# llm.invoke() call, each attempt re-waiting up to the full timeout — so a
# genuinely dead connection took 3x _REQUEST_TIMEOUT (confirmed empirically:
# timeout=10 took 31s to raise, not 10s) before our own call_with_retry ever
# saw an exception. We already do capacity-aware retries with visible
# logging in utils/retry.py — the SDK's hidden retry layer only adds an
# invisible multiplier on top of it, which is what turned a ~2min timeout
# into real-world 15-40min hangs.
_REQUEST_TIMEOUT = 120
_SDK_MAX_RETRIES = 0


def get_llm(model: str, params: dict = DEFAULT_PARAMS):
    if MODELS.get(model) == "huggingface":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=HF_MODEL_MAP[model],
            base_url=HF_API_URL,
            api_key=HF_API_KEY,
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 2048),
            timeout=_REQUEST_TIMEOUT,
            max_retries=_SDK_MAX_RETRIES,
        )
    if MODELS.get(model) == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            base_url=GROQ_API_URL,
            api_key=GROQ_API_KEY,
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 2048),
            timeout=_REQUEST_TIMEOUT,
            max_retries=_SDK_MAX_RETRIES,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        base_url=LAB_API_URL,
        api_key=LAB_API_KEY,
        temperature=params.get("temperature", 0.0),
        max_tokens=params.get("max_tokens", 2048),
        timeout=_REQUEST_TIMEOUT,
        max_retries=_SDK_MAX_RETRIES,
    )
