"""Batch API wiring, against a fake Anthropic client: no network, no API key."""
import itertools
import json
import time
from types import SimpleNamespace

import pytest
from anthropic.types import Message
from langchain_core.tools import tool

from agent.batching import BatchChatAnthropic, BatchError, BatchScheduler
from agent.schemas import Plan
from observability.usage import UsageTracker


def message(content: list[dict], input_tokens: int = 100, output_tokens: int = 20) -> Message:
    return Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5", "content": content,
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })


class FakeBatches:
    """Records submitted batches; each ends after `polls_until_done` polls with whatever `respond` returns."""

    def __init__(self, respond, polls_until_done: int = 1, fail_create: bool = False):
        self.respond, self.polls_until_done, self.fail_create = respond, polls_until_done, fail_create
        self.created: list[list[dict]] = []
        self.cancelled: list[str] = []
        self._polls: dict[str, int] = {}
        self._ids = itertools.count()

    def create(self, requests):
        if self.fail_create:
            raise ConnectionError("network down")
        batch_id = f"batch_{next(self._ids)}"
        self.created.append(list(requests))
        self._polls[batch_id] = 0
        return SimpleNamespace(id=batch_id)

    def retrieve(self, batch_id):
        self._polls[batch_id] += 1
        ended = self._polls[batch_id] >= self.polls_until_done
        return SimpleNamespace(processing_status="ended" if ended else "in_progress")

    def results(self, batch_id):
        requests = self.created[int(batch_id.split("_")[1])]
        return [SimpleNamespace(custom_id=r["custom_id"], result=self.respond(r)) for r in requests]

    def cancel(self, batch_id):
        self.cancelled.append(batch_id)


def client(batches: FakeBatches):
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def succeeded(text="ok"):
    return lambda request: SimpleNamespace(type="succeeded", message=message([{"type": "text", "text": text}]))


def scheduler(batches, **kwargs) -> BatchScheduler:
    return BatchScheduler(client(batches), window_seconds=kwargs.pop("window", 0.05), poll_seconds=0.01, **kwargs)


def test_requests_in_the_same_window_share_one_batch():
    batches = FakeBatches(succeeded())
    s = scheduler(batches)
    futures = [s.submit({"model": "m", "messages": [{"role": "user", "content": str(i)}]}) for i in range(3)]
    assert [f.result(timeout=5).content[0].text for f in futures] == ["ok", "ok", "ok"]
    assert len(batches.created) == 1 and len(batches.created[0]) == 3
    assert len({r["custom_id"] for r in batches.created[0]}) == 3


def test_each_caller_gets_its_own_result():
    batches = FakeBatches(lambda r: SimpleNamespace(
        type="succeeded", message=message([{"type": "text", "text": r["params"]["messages"][0]["content"]}])))
    s = scheduler(batches)
    futures = {i: s.submit({"messages": [{"role": "user", "content": f"q{i}"}]}) for i in range(4)}
    assert {i: f.result(timeout=5).content[0].text for i, f in futures.items()} == {i: f"q{i}" for i in range(4)}


def test_failed_requests_raise_for_that_caller_only():
    batches = FakeBatches(lambda r: SimpleNamespace(type="errored", error={"type": "overloaded_error"})
                          if r["params"]["bad"] else succeeded()(r))
    s = scheduler(batches)
    good, bad = s.submit({"bad": False}), s.submit({"bad": True})
    assert good.result(timeout=5).content[0].text == "ok"
    with pytest.raises(BatchError, match="errored"):
        bad.result(timeout=5)


def test_create_failure_and_timeouts_fail_cleanly():
    with pytest.raises(BatchError, match="could not create batch"):
        scheduler(FakeBatches(succeeded(), fail_create=True)).submit({}).result(timeout=5)
    slow = FakeBatches(succeeded(), polls_until_done=10_000)
    s = scheduler(slow, max_wait_seconds=0.1)
    with pytest.raises(BatchError, match="still running"):
        s.submit({}).result(timeout=5)
    assert slow.cancelled == ["batch_0"]


@pytest.fixture
def batch_llm(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used")

    def make(respond) -> tuple[BatchChatAnthropic, FakeBatches]:
        llm = BatchChatAnthropic(model="claude-sonnet-5", max_tokens=1000)
        batches = FakeBatches(respond)
        llm._scheduler = scheduler(batches, window=0.01)
        return llm, batches
    return make


def test_plain_call_goes_through_a_batch_and_reports_usage(batch_llm):
    llm, batches = batch_llm(succeeded("hello"))
    tracker = UsageTracker()
    reply = llm.invoke("hi", config={"callbacks": [tracker]})
    assert reply.text == "hello"
    assert reply.response_metadata["batch"] is True and reply.usage_metadata["input_tokens"] == 100
    (request,) = batches.created[0]
    assert request["params"]["model"] == "claude-sonnet-5" and "stream" not in request["params"]
    assert tracker.cost_usd() == pytest.approx((100 * 2.0 + 20 * 10.0) / 1e6 * 0.5)  # half price


def test_tool_calls_survive_the_batch_round_trip(batch_llm):
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return ""

    llm, batches = batch_llm(lambda r: SimpleNamespace(type="succeeded", message=message(
        [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}}])))
    reply = llm.bind_tools([read_file]).invoke("read a.py")
    assert reply.tool_calls == [{"name": "read_file", "args": {"path": "a.py"}, "id": "toolu_1", "type": "tool_call"}]
    assert batches.created[0][0]["params"]["tools"][0]["name"] == "read_file"


def test_structured_output_survives_the_batch_round_trip(batch_llm):
    plan = {"problem": "p", "relevant_files": ["a.py"], "implementation_steps": [], "tests_needed": [], "risks": []}
    llm, batches = batch_llm(succeeded(json.dumps(plan)))
    result = llm.with_structured_output(Plan, method="json_schema").invoke("plan it")
    assert result == Plan(**plan)
    assert "output_config" in batches.created[0][0]["params"]


async def _async_call(llm):
    return await llm.ainvoke("hi")


def test_async_calls_wait_for_their_batch(batch_llm):
    import asyncio

    llm, _ = batch_llm(succeeded("async ok"))
    started = time.monotonic()
    assert asyncio.run(_async_call(llm)).text == "async ok"
    assert time.monotonic() - started < 5


def test_batch_mode_needs_anthropic():
    from agent.llm import LLMSettings, get_chat_model

    with pytest.raises(ValueError, match="anthropic provider"):
        get_chat_model(LLMSettings(provider="openai", model="x"), batch=True)
