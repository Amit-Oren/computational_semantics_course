import os
import logging
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict

load_dotenv()

LAB_API_KEY = os.getenv("LAB_API_KEY", "")
LAB_API_URL = os.getenv("LAB_API_URL", "http://100.110.96.82:8000/chat")

MODELS = {
    "llama3.1-8b":       "open_source",
    "llama3.1-70b":      "open_source",
    "qwen2.5-32b":       "open_source",
    "qwen2.5-72b":       "open_source",
    "qwen3.6-35b":       "open_source",
    "qwen3.6-35b-ablit": "open_source",
    "gpt-oss-20b":       "open_source",
    "gemma4-31b":        "open_source",
    "nemotron-70b":      "open_source",
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


def setup_logger(experiment: str, model: str) -> logging.Logger:
    from datetime import datetime
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"{experiment}_{model}_{timestamp}.log")

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


def get_llm(model: str, params: dict = DEFAULT_PARAMS):
    from open_source_llm import OpenSourceChatModel
    return OpenSourceChatModel(
        model=model,
        api_url=LAB_API_URL,
        api_key=LAB_API_KEY,
        temperature=params.get("temperature", 0.0),
        max_tokens=params.get("max_tokens", 512),
    )
