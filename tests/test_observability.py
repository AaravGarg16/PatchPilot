import asyncio
import uuid
from typing import ClassVar

import langfuse
import langfuse.langchain
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent.graph import build_graph
from agent.llm import LLMSettings, get_chat_model
from agent.run import run_issue
from agent.schemas import Evaluation, Plan
from agent.state import PatchPilotState
from observability import langfuse as tracing
from observability.pricing import cost_usd
from observability.usage import UsageTracker
from tests.fakes import (
    FAKE_MODEL,
    FIX_PAGINATE,
    FIX_TOTAL_PAGES,
    FIXTURE_REPO,
    FakeLLM,
    done,
    tool_call,
)
from tests.test_graph import FINISH, PLAN, retry


def llm_result(model: str, input_tokens: int, output_tokens: int) -> LLMResult:
    message = AIMessage(
        "x",
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens},
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_usage_tracker_sums_tokens_and_prices_them():
    tracker = UsageTracker()
    tracker.on_llm_end(llm_result("claude-sonnet-5", 1_000_000, 0))
    tracker.on_llm_end(llm_result("claude-haiku-4-5", 0, 1_000_000))
    assert (tracker.calls, tracker.input_tokens, tracker.output_tokens) == (2, 1_000_000, 1_000_000)
    assert tracker.cost_usd() == pytest.approx(2.00 + 5.00)


def test_unknown_model_cost_is_none_not_zero():
    tracker = UsageTracker()
    tracker.on_llm_end(llm_result("some-new-model", 10, 10))
    assert tracker.cost_usd() is None
    assert cost_usd("claude-sonnet-5", 0, 0) == 0


def test_tracing_is_a_no_op_without_keys(monkeypatch):
    monkeypatch.setattr(tracing, "load_dotenv", lambda: None)
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with tracing.trace_run("run1", input={}, metadata={}) as trace:
        assert trace.callbacks == [] and trace.trace_id is None
        trace.finish(output={}, scores={"success": 1.0})  # must not raise


def test_tracing_switch_overrides_keys(monkeypatch):
    monkeypatch.setattr(tracing, "load_dotenv", lambda: None)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    assert not tracing.langfuse_enabled()
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    assert tracing.langfuse_enabled()


class RecordingHandler(BaseCallbackHandler):
    """Stands in for Langfuse's CallbackHandler and records what it would have traced."""

    def __init__(self):
        self.llm_runs: list[str] = []
        self.tool_runs: list[str] = []
        self.chain_runs: list[str] = []

    def on_chat_model_start(self, serialized, messages, *, name=None, **kwargs):
        self.llm_runs.append(name or "")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_runs.append(serialized.get("name", ""))

    def on_chain_start(self, serialized, inputs, *, name=None, **kwargs):
        self.chain_runs.append(name or "")


class FakeSpan:
    def __init__(self):
        self.updates, self.scores = [], {}

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def score_trace(self, *, name, value):
        self.scores[name] = value


class FakeLangfuse:
    instances: ClassVar[list["FakeLangfuse"]] = []

    def __init__(self):
        self.span = FakeSpan()
        self.flushed = False
        self.span_kwargs = None
        FakeLangfuse.instances.append(self)

    def start_as_current_observation(self, **kwargs):
        from contextlib import nullcontext

        self.span_kwargs = kwargs
        return nullcontext(self.span)

    def get_current_trace_id(self):
        return "trace-123"

    def flush(self):
        self.flushed = True


@pytest.fixture
def fake_langfuse(monkeypatch):
    from contextlib import nullcontext

    handlers: list[RecordingHandler] = []
    propagated: list[dict] = []
    FakeLangfuse.instances.clear()
    monkeypatch.setattr(tracing, "load_dotenv", lambda: None)
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    monkeypatch.setattr(langfuse, "propagate_attributes", lambda **kw: propagated.append(kw) or nullcontext())
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", lambda: handlers.append(RecordingHandler()) or handlers[-1])
    return handlers, propagated


def test_trace_flushes_even_when_the_run_fails(fake_langfuse):
    with pytest.raises(RuntimeError), tracing.trace_run("run1", input={}, metadata={}):
        raise RuntimeError("boom")
    assert FakeLangfuse.instances[0].flushed


def test_every_llm_call_tool_call_and_retry_is_traced(fake_langfuse, retriever, sandbox_image, tmp_path):
    handlers, propagated = fake_langfuse
    llm = FakeLLM(
        {Plan: [PLAN], Evaluation: [retry("total_pages still floors"), FINISH]},
        executor_turns=[tool_call("read_file", path="shop/pagination.py"), FIX_PAGINATE, done(), FIX_TOTAL_PAGES, done()],
    )

    report = asyncio.run(run_issue(FIXTURE_REPO, "Pagination skips the first page", llm=llm, model_name=FAKE_MODEL,
                                   retriever=retriever, workspace_root=str(tmp_path)))

    (handler,) = handlers
    assert report.trace_id == "trace-123"
    assert propagated[0]["session_id"] == report.run_id
    # Every LLM call, named so retries are told apart in the trace.
    assert handler.llm_runs == [
        "planner",
        "executor attempt 1", "executor attempt 1", "executor attempt 1",
        "evaluator attempt 1",
        "executor attempt 2", "executor attempt 2",
        "evaluator attempt 2",
    ]
    assert handler.tool_runs == ["read_file", "edit_file", "edit_file"]
    # Graph nodes are spans too, and the retry shows up as a second pass through implement.
    assert handler.chain_runs.count("implement") == 2
    assert handler.chain_runs.count("refresh_context") == 1

    span = FakeLangfuse.instances[0].span
    assert span.scores == {"success": 1.0, "attempts": 2, "tool_errors": 0}
    assert span.updates[-1]["output"]["success"] is True
    assert FakeLangfuse.instances[0].flushed


@pytest.fixture
def offline_langfuse(monkeypatch):
    """The real Langfuse client and CallbackHandler, exporting spans to memory instead of the network."""
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")  # opt back in; spans only ever reach the in-memory exporter
    exporter = InMemorySpanExporter()
    # Langfuse keeps one client per public key for the whole process, so each test needs its own key.
    public_key = f"pk-offline-{uuid.uuid4().hex}"
    client = langfuse.Langfuse(public_key=public_key, secret_key="sk-offline-test",
                               base_url="http://127.0.0.1:9", span_exporter=exporter, tracing_enabled=True)
    yield client, exporter, public_key
    client.shutdown()


def generation_models(client, exporter) -> list:
    client.flush()
    return [
        span.attributes.get("langfuse.observation.model.name")
        for span in exporter.get_finished_spans()
        if span.attributes.get("langfuse.observation.type") == "generation"
    ]


def test_langfuse_records_the_model_for_real_chat_anthropic(offline_langfuse, monkeypatch, caplog):
    """Feeds Langfuse exactly what a real ChatAnthropic call reports, stopping before any HTTP request."""
    client, exporter, public_key = offline_langfuse
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used")

    class StopBeforeRequest(Exception):
        pass

    class Capture(BaseCallbackHandler):
        raise_error = True

        def on_chat_model_start(self, serialized, messages, **kwargs):
            self.args = (serialized, kwargs)
            raise StopBeforeRequest

    capture = Capture()
    with pytest.raises(StopBeforeRequest):
        get_chat_model(LLMSettings(provider="anthropic", model="claude-sonnet-5")).invoke(
            "hi", config={"callbacks": [capture]}
        )
    serialized, kwargs = capture.args

    handler = langfuse.langchain.CallbackHandler(public_key=public_key)
    run_id = kwargs.pop("run_id")
    kwargs.pop("parent_run_id", None)
    handler.on_chat_model_start(serialized, [[HumanMessage("hi")]], run_id=run_id, **kwargs)
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage("ok"))]]), run_id=run_id)

    assert generation_models(client, exporter) == ["claude-sonnet-5"]
    assert "not able to parse the LLM model" not in caplog.text


def test_every_generation_in_a_run_records_its_model(offline_langfuse, retriever, sandbox_image, tmp_path, caplog):
    client, exporter, public_key = offline_langfuse
    handler = langfuse.langchain.CallbackHandler(public_key=public_key)
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, FIX_TOTAL_PAGES, done()])
    graph = build_graph(llm, retriever, workspace_root=str(tmp_path))

    asyncio.run(graph.ainvoke(PatchPilotState(repo_id=FIXTURE_REPO, issue="Pagination skips the first page"),
                              config={"callbacks": [handler]}))

    assert generation_models(client, exporter) == [FAKE_MODEL] * 5
    assert "not able to parse the LLM model" not in caplog.text
