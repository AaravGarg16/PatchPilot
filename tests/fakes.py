from pathlib import Path

FIXTURE_REPO = str(Path(__file__).resolve().parents[1] / "examples" / "pagination_repo")

FIXED_PAGINATION = '''import math


def total_pages(total_items: int, per_page: int) -> int:
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    return math.ceil(total_items / per_page)


def paginate(items: list, page: int, per_page: int) -> list:
    """Return the items on a 1-indexed page."""
    if page < 1:
        raise ValueError("page must be at least 1")
    start = (page - 1) * per_page
    return items[start:start + per_page]
'''

# Fixes paginate but leaves total_pages broken, so tests still fail.
HALF_FIXED_PAGINATION = FIXED_PAGINATION.replace(
    "math.ceil(total_items / per_page)", "total_items // per_page"
)


class FakeLLM:
    """Stands in for the chat model. Returns queued responses per schema; the last one repeats."""

    def __init__(self, responses: dict):
        self.responses = {schema: list(queue) for schema, queue in responses.items()}
        self.calls: list[tuple[type, list]] = []

    def with_structured_output(self, schema, **kwargs):
        return _Bound(self, schema)

    def prompts_for(self, schema) -> list[str]:
        return [messages[-1][1] for s, messages in self.calls if s is schema]


class _Bound:
    def __init__(self, llm: FakeLLM, schema):
        self.llm = llm
        self.schema = schema

    def invoke(self, messages):
        self.llm.calls.append((self.schema, messages))
        queue = self.llm.responses[self.schema]
        return queue.pop(0) if len(queue) > 1 else queue[0]
