import itertools
from pathlib import Path

from langchain_core.messages import AIMessage

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

_ids = itertools.count()


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def done(summary: str = "Fixed it.") -> AIMessage:
    return AIMessage(content=summary)


# The two halves of the real fix, as edit_file calls against the fixture repo.
FIX_PAGINATE = tool_call(
    "edit_file", path="shop/pagination.py", old_string="start = page * per_page", new_string="start = (page - 1) * per_page"
)
FIX_TOTAL_PAGES = tool_call(
    "edit_file", path="shop/pagination.py",
    old_string="return total_items // per_page", new_string="return -(-total_items // per_page)",
)


class FakeLLM:
    """Stands in for the chat model, so tests never call the API.

    Structured output: queued responses per schema (the last one repeats).
    Tool calling: one flat queue of AIMessages; an attempt ends at the first message without tool calls.
    """

    def __init__(self, structured: dict, executor_turns: list[AIMessage] | None = None):
        self.structured = {schema: list(queue) for schema, queue in structured.items()}
        self.executor_turns = list(executor_turns or [])
        self.structured_calls: list[tuple[type, list]] = []
        self.executor_calls: list[list] = []
        self.bound_tool_names: list[str] = []

    def with_structured_output(self, schema, **kwargs):
        return _Structured(self, schema)

    def bind_tools(self, tools):
        self.bound_tool_names = sorted(tool.name for tool in tools)
        return _ToolCaller(self)

    def prompts_for(self, schema) -> list[str]:
        return [messages[-1][1] for s, messages in self.structured_calls if s is schema]

    def executor_task_prompts(self) -> list[str]:
        """The task message that opened each executor attempt."""
        return [msgs[1].content for msgs in self.executor_calls if len(msgs) == 2]


class _Structured:
    def __init__(self, llm: FakeLLM, schema):
        self.llm, self.schema = llm, schema

    def invoke(self, messages):
        self.llm.structured_calls.append((self.schema, messages))
        queue = self.llm.structured[self.schema]
        return queue.pop(0) if len(queue) > 1 else queue[0]


class _ToolCaller:
    def __init__(self, llm: FakeLLM):
        self.llm = llm

    async def ainvoke(self, messages):
        self.llm.executor_calls.append(list(messages))
        if not self.llm.executor_turns:
            raise AssertionError("executor asked for more turns than the test scripted")
        return self.llm.executor_turns.pop(0)
