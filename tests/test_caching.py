"""Prompt caching: where the executor puts cache markers, and how cached tokens are priced. No API calls."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agent.executor import cache_breakpoints
from agent.llm import CACHE_CONTROL, supports_prompt_caching
from agent.schemas import Evaluation, Plan
from evals.estimate import estimate
from evals.metrics import summarize
from evals.results import TaskResult
from observability.pricing import cost_usd
from observability.usage import UsageTracker, split_input
from tests.fakes import FIX_PAGINATE, FIX_TOTAL_PAGES, FakeLLM, done, tool_call
from tests.test_graph import FINISH, PLAN, retry

CONVERSATION = [
    SystemMessage("system prompt"),
    HumanMessage("issue, plan and retrieved code"),
    AIMessage("", tool_calls=[{"name": "read_file", "args": {}, "id": "c1"}]),
    ToolMessage("file contents", tool_call_id="c1"),
]


def test_the_task_message_carries_the_stable_prefix_marker():
    marked, kwargs = cache_breakpoints(CONVERSATION, cache_tail=False)
    assert marked[1].content == [{"type": "text", "text": "issue, plan and retrieved code", "cache_control": CACHE_CONTROL}]
    assert marked[0] is CONVERSATION[0] and marked[2:] == CONVERSATION[2:]
    assert CONVERSATION[1].content == "issue, plan and retrieved code"  # the stored history is never changed
    assert kwargs == {}
    assert cache_breakpoints(CONVERSATION, cache_tail=True)[1] == {"cache_control": CACHE_CONTROL}


def test_markers_reach_the_real_anthropic_request(monkeypatch):
    """Built with ChatAnthropic's own payload code; nothing is sent."""
    from langchain_anthropic import ChatAnthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used")
    llm = ChatAnthropic(model="claude-sonnet-5", max_tokens=1000)
    assert supports_prompt_caching(llm)
    marked, kwargs = cache_breakpoints(CONVERSATION, cache_tail=True)
    payload = llm._get_request_payload(marked, **kwargs)
    assert payload["cache_control"] == CACHE_CONTROL  # automatic caching of the growing tail
    assert payload["messages"][0]["content"][0]["cache_control"] == CACHE_CONTROL  # explicit stable-prefix marker
    assert payload["system"] == "system prompt" or payload["system"][0]["text"] == "system prompt"


def test_other_providers_get_no_markers():
    assert not supports_prompt_caching(FakeLLM({}))
    assert supports_prompt_caching(FakeLLM({}, caching=True))


def test_executor_caches_the_tail_only_until_it_starts_trimming(run_report):
    reads = [tool_call("read_file", path=p) for p in ("tests/test_pagination.py", "shop/formatting.py", "README.md")]
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]},
                  executor_turns=[*reads, FIX_PAGINATE, FIX_TOTAL_PAGES, done()], caching=True)

    state = run_report(llm, context_budget=3000).state

    assert state.elided_tool_outputs >= 1
    assert all(call[1].content[0]["cache_control"] == CACHE_CONTROL for call in llm.executor_calls)
    tail_cached = [bool(kwargs) for kwargs in llm.executor_kwargs]
    first_trim = next(i for i, call in enumerate(llm.executor_calls)
                      if any(isinstance(m, ToolMessage) and str(m.content).startswith("[elided") for m in call))
    assert all(tail_cached[:first_trim]) and not any(tail_cached[first_trim:])


def test_full_history_always_caches_the_tail(run_report):
    from evals.configs import CONFIGS

    llm = FakeLLM({Plan: [PLAN], Evaluation: [retry("total_pages still floors"), FINISH]},
                  executor_turns=[FIX_PAGINATE, done(), FIX_TOTAL_PAGES, done()], caching=True)

    run_report(llm, agent_config=CONFIGS["full-history"])

    assert len(llm.executor_kwargs) == 4 and all(kwargs == {"cache_control": CACHE_CONTROL} for kwargs in llm.executor_kwargs)


def test_no_markers_without_caching_support(run_report):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, FIX_TOTAL_PAGES, done()])
    run_report(llm)
    assert all(isinstance(call[1].content, str) for call in llm.executor_calls)
    assert all(kwargs == {} for kwargs in llm.executor_kwargs)


# --- pricing ---------------------------------------------------------------------------------------------------------


def test_cache_reads_and_writes_are_priced_at_their_rates():
    # Sonnet 5: $2 / MTok input. Reads 0.1x, 5-minute writes 1.25x, 1-hour writes 2x.
    assert cost_usd("claude-sonnet-5", 1_000_000, 0) == pytest.approx(2.0)
    assert cost_usd("claude-sonnet-5", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.2)
    assert cost_usd("claude-sonnet-5", 0, 0, cache_write_tokens=1_000_000) == pytest.approx(2.5)
    assert cost_usd("claude-sonnet-5", 0, 0, cache_write_1h_tokens=1_000_000) == pytest.approx(4.0)
    assert cost_usd("claude-sonnet-5", 0, 0, batch=True, cache_read_tokens=1_000_000) == pytest.approx(0.1)


def test_usage_split_follows_langchain_metadata():
    # LangChain's input_tokens already includes cache reads and writes.
    usage = {"input_tokens": 10_000, "output_tokens": 50,
             "input_token_details": {"cache_read": 7_000, "cache_creation": 0, "ephemeral_5m_input_tokens": 2_000}}
    assert split_input(usage) == (1_000, 7_000, 2_000, 0)
    generic = {"input_tokens": 10_000, "input_token_details": {"cache_read": 7_000, "cache_creation": 2_000}}
    assert split_input(generic) == (1_000, 7_000, 2_000, 0)
    assert split_input({"input_tokens": 500}) == (500, 0, 0, 0)


def test_tracker_prices_a_cached_call():
    message = AIMessage("x", response_metadata={"model_name": "claude-sonnet-5"}, usage_metadata={
        "input_tokens": 10_000, "output_tokens": 100, "total_tokens": 10_100,
        "input_token_details": {"cache_read": 7_000, "ephemeral_5m_input_tokens": 2_000}})
    tracker = UsageTracker()
    tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    assert (tracker.input_tokens, tracker.cache_read_tokens, tracker.cache_write_tokens) == (10_000, 7_000, 2_000)
    expected = (1_000 * 1 + 7_000 * 0.1 + 2_000 * 1.25) * 2 / 1e6 + 100 * 10 / 1e6
    assert tracker.cost_usd() == pytest.approx(expected)


def test_cached_input_share_metric():
    results = [TaskResult(instance_id=f"t{i}", source="swebench", config="full", status="done", llm_calls=1,
                          input_tokens=1_000, cache_read_tokens=read) for i, read in enumerate((600, 200))]
    assert summarize("full", results).cached_input_share == pytest.approx(0.4)


def test_estimate_prices_caching_below_no_caching():
    rows = estimate("claude-sonnet-5")
    assert all(r["cached"] < r["standard"] for r in rows)
    measured = {"full": (100_000, 1_000, 3, (0.9, 0.05))}
    headline = estimate("claude-sonnet-5", measured=measured)[1]  # only new `full` runs
    expected = headline["new_runs"] * cost_usd("claude-sonnet-5", 5_000, 1_000, cache_read_tokens=90_000,
                                                cache_write_tokens=5_000)
    assert headline["cached"] == pytest.approx(expected)
