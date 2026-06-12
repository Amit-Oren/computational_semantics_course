"""LangChain-compatible wrapper for the lab Tailscale REST API."""
import json
import logging
import re
import requests
from typing import Any, List, Optional, Type
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


class _StructuredOutputWrapper:
    def __init__(self, llm, schema: Type[BaseModel]):
        self._llm    = llm
        self._schema = schema

    def invoke(self, messages: list, **kwargs) -> BaseModel:
        augmented = list(messages)
        fields = list(self._schema.model_json_schema()["properties"].keys())
        json_instruction = (
            f"\nRespond with a JSON object only — no explanation, no markdown fences, no schema definition. "
            f"Fill in actual values for these fields: {fields}. "
            f"Field types for reference: {self._schema.model_json_schema()['properties']}"
        )
        last = augmented[-1]
        new_content = (last["content"] if isinstance(last, dict) else last.content) + json_instruction
        augmented[-1] = HumanMessage(content=new_content)

        result = self._llm.invoke(augmented)
        text = result.content if hasattr(result, "content") else str(result)
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        try:
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if not match:
                logger.warning("No JSON object found. Raw text: %r", text[:300])
                return None
            return self._schema(**json.loads(match.group()))
        except Exception as e:
            logger.warning("Structured output parse failed (%s). Text: %r", e, clean[:300])
            return None


class OpenSourceChatModel(BaseChatModel):
    model:       str
    api_url:     str
    api_key:     str
    temperature: float = 0.0
    max_tokens:  int   = 512
    timeout:     int   = 600

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  **kwargs: Any) -> ChatResult:
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), None)
        prompt = "\n".join(m.content for m in messages if not isinstance(m, SystemMessage))

        response = requests.post(
            self.api_url,
            headers={"x-api-key": self.api_key},
            json={"model": self.model, "prompt": prompt, "system": system},
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        text = body["text"]

        usage = body.get("usage", {})
        input_tokens  = usage.get("input_tokens")  or usage.get("prompt_tokens",     0)
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)

        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=text, usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            })
        )])

    def with_structured_output(self, schema: Type[BaseModel], **kwargs) -> _StructuredOutputWrapper:
        return _StructuredOutputWrapper(self, schema)

    @property
    def _llm_type(self) -> str:
        return "open_source"
