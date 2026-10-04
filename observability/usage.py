import threading
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from observability.pricing import cost_usd


class UsageTracker(BaseCallbackHandler):
    """Sums token usage over every LLM call in a run, independent of Langfuse, so metrics exist even without tracing."""

    def __init__(self):
        self._lock = threading.Lock()  # sync callbacks can fire from LangGraph's worker threads
        self.calls = 0
        self.by_model: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.prompt_tokens: list[int] = []  # input tokens per call, for context utilization

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                model = (getattr(message, "response_metadata", None) or {}).get("model_name", "unknown")
                with self._lock:
                    self.calls += 1
                    self.by_model[model][0] += usage.get("input_tokens", 0)
                    self.prompt_tokens.append(usage.get("input_tokens", 0))
                    self.by_model[model][1] += usage.get("output_tokens", 0)

    @property
    def input_tokens(self) -> int:
        return sum(i for i, _ in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(o for _, o in self.by_model.values())

    def cost_usd(self) -> float | None:
        """None if any model's price is unknown, rather than silently under-reporting."""
        costs = [cost_usd(model, i, o) for model, (i, o) in self.by_model.items()]
        return None if any(c is None for c in costs) else sum(costs)
