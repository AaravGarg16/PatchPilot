import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-5"
# Explicit cap so cost per call is predictable; whole-file edits still fit comfortably.
MAX_TOKENS = 16000


# Prompt caching: the 5-minute TTL. Executor turns are seconds apart, and each read refreshes the entry, so the
# 1-hour TTL would only double the write price.
CACHE_CONTROL = {"type": "ephemeral"}


def supports_prompt_caching(llm) -> bool:
    """Only Anthropic models take cache markers; other providers would reject or ignore them."""
    from langchain_anthropic import ChatAnthropic

    return isinstance(llm, ChatAnthropic) or bool(getattr(llm, "supports_prompt_caching", False))


class LLMSettings(BaseModel):
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL


def load_llm_settings() -> LLMSettings:
    load_dotenv()
    return LLMSettings(
        provider=os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER,
        model=os.getenv("LLM_MODEL") or DEFAULT_MODEL,
    )


def get_chat_model(settings: LLMSettings | None = None, batch: bool = False):
    """batch=True sends every call through Anthropic's Batch API: half the price, minutes of latency per call."""
    settings = settings or load_llm_settings()
    if batch:
        if settings.provider != "anthropic":
            raise ValueError(f"batch mode needs the anthropic provider, not {settings.provider!r}")
        from agent.batching import BatchChatAnthropic

        return BatchChatAnthropic(model=settings.model, max_tokens=MAX_TOKENS)
    return init_chat_model(settings.model, model_provider=settings.provider, max_tokens=MAX_TOKENS)
