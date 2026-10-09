import threading
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

from observability.pricing import cost_usd


class Tokens(BaseModel):
    uncached: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    output: int = 0

    @property
    def input(self) -> int:
        return self.uncached + self.cache_read + self.cache_write_5m + self.cache_write_1h


def split_input(usage: dict) -> tuple[int, int, int, int]:
    """(uncached, cache read, 5-minute write, 1-hour write) from LangChain usage metadata.

    LangChain's input_tokens is the total, cached tokens included; the split is in input_token_details.
    """
    details = usage.get("input_token_details") or {}
    read = details.get("cache_read") or 0
    write_5m = details.get("ephemeral_5m_input_tokens") or 0
    write_1h = details.get("ephemeral_1h_input_tokens") or 0
    if not (write_5m or write_1h):
        write_5m = details.get("cache_creation") or 0  # no TTL breakdown: the default TTL is 5 minutes
    uncached = max((usage.get("input_tokens") or 0) - read - write_5m - write_1h, 0)
    return uncached, read, write_5m, write_1h


class UsageTracker(BaseCallbackHandler):
    """Sums token usage over every LLM call in a run, independent of Langfuse, so metrics exist even without tracing.

    Input is kept split into uncached, cache-read and cache-write tokens, because each is billed differently.
    """

    def __init__(self):
        self._lock = threading.Lock()  # sync callbacks can fire from LangGraph's worker threads
        self.calls = 0
        # (model, batch) -> tokens; batch calls are priced at the batch discount.
        self.by_model: dict[tuple[str, bool], Tokens] = defaultdict(Tokens)
        self.prompt_tokens: list[int] = []  # total input tokens per call, for context utilization

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                metadata = getattr(message, "response_metadata", None) or {}
                key = (metadata.get("model_name", "unknown"), bool(metadata.get("batch")))
                uncached, read, write_5m, write_1h = split_input(usage)
                with self._lock:
                    self.calls += 1
                    tokens = self.by_model[key]
                    tokens.uncached += uncached
                    tokens.cache_read += read
                    tokens.cache_write_5m += write_5m
                    tokens.cache_write_1h += write_1h
                    tokens.output += usage.get("output_tokens", 0)
                    self.prompt_tokens.append(usage.get("input_tokens", 0))

    @property
    def input_tokens(self) -> int:
        return sum(t.input for t in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(t.output for t in self.by_model.values())

    @property
    def cache_read_tokens(self) -> int:
        return sum(t.cache_read for t in self.by_model.values())

    @property
    def cache_write_tokens(self) -> int:
        return sum(t.cache_write_5m + t.cache_write_1h for t in self.by_model.values())

    def cost_usd(self) -> float | None:
        """None if any model's price is unknown, rather than silently under-reporting."""
        costs = [
            cost_usd(model, t.uncached, t.output, batch, cache_read_tokens=t.cache_read,
                     cache_write_tokens=t.cache_write_5m, cache_write_1h_tokens=t.cache_write_1h)
            for (model, batch), t in self.by_model.items()
        ]
        return None if any(c is None for c in costs) else sum(costs)
