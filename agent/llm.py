import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-5"


class LLMSettings(BaseModel):
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL


def load_llm_settings() -> LLMSettings:
    load_dotenv()
    return LLMSettings(
        provider=os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER,
        model=os.getenv("LLM_MODEL") or DEFAULT_MODEL,
    )


def get_chat_model(settings: LLMSettings | None = None):
    settings = settings or load_llm_settings()
    return init_chat_model(settings.model, model_provider=settings.provider)
