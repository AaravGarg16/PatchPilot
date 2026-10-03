from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.context import (
    ContextBudget,
    build_code_context,
    compress_tool_outputs,
    estimate_tokens,
    fit,
    render_failures,
)
from agent.prompts import evaluator, executor, planner
from agent.state import FailureRecord
from retrieval.retriever import ScoredChunk
from tests.fakes import FIXTURE_REPO


def chunk(id_: int, path: str, content: str, score: float = 0.1) -> ScoredChunk:
    return ScoredChunk(id=id_, path=path, language="python", symbol=f"s{id_}", chunk_type="function",
                       start_line=1, end_line=1, content=content, score=score)


def failure(attempt: int, problem: str, output: str = "1 failed") -> FailureRecord:
    return FailureRecord(attempt=attempt, tests_passed=False, test_output=output,
                         lint_passed=True, lint_output="", problems=[problem])


def test_budget_follows_spec_shares():
    budget = ContextBudget(total_tokens=10_000)
    assert [budget.tokens(s) for s in ("system", "task", "plan", "code", "failures", "tool_outputs")] == [
        1000, 1000, 1000, 4500, 1500, 1000,
    ]


def test_system_prompts_fit_their_share():
    limit = ContextBudget().tokens("system")
    assert all(estimate_tokens(p.SYSTEM) <= limit for p in (planner, executor, evaluator))


def test_fit_keeps_head_or_tail():
    text = "start " + "x" * 1000 + " end"
    assert fit(text, 10).startswith("start") and "trimmed" in fit(text, 10)
    assert fit(text, 10, keep="tail").endswith(" end")
    assert fit("short", 10) == "short"


def test_code_context_prefers_whole_selected_files():
    context = build_code_context(FIXTURE_REPO, ["shop/pagination.py"], [chunk(1, "shop/pagination.py", "stale")], 2000)
    assert "shop/pagination.py (whole file)" in context
    assert "stale" not in context  # chunks of a file shown whole are not repeated


def test_code_context_falls_back_to_chunks_and_respects_budget():
    big = chunk(1, "shop/pagination.py", "def paginate(): ...")
    filler = [chunk(i, "other.py", "y" * 400, score=0.01) for i in range(2, 30)]
    whole_file = estimate_tokens((Path(FIXTURE_REPO) / "shop/pagination.py").read_text())
    budget = whole_file // 2

    context = build_code_context(FIXTURE_REPO, ["shop/pagination.py"], [big, *filler], max_tokens=budget)

    assert "whole file" not in context
    assert "def paginate(): ..." in context
    assert "y" * 400 not in context
    assert estimate_tokens(context) <= budget


def test_tool_outputs_beyond_budget_are_elided_oldest_first():
    messages = [HumanMessage("task")]
    for i in range(4):
        messages += [AIMessage("", tool_calls=[{"name": "read_file", "args": {}, "id": f"c{i}"}]),
                     ToolMessage("z" * 400, tool_call_id=f"c{i}", name="read_file")]

    elided = compress_tool_outputs(messages, max_tokens=250)

    contents = [m.content for m in messages if isinstance(m, ToolMessage)]
    assert elided == 2
    assert contents[0].startswith("[elided read_file") and contents[1].startswith("[elided")
    assert contents[2] == contents[3] == "z" * 400
    assert compress_tool_outputs(messages, max_tokens=250) == 0  # already elided outputs are not counted twice


def test_the_latest_tool_output_survives_even_when_over_budget():
    messages = [ToolMessage("z" * 4000, tool_call_id="c", name="read_file")]
    assert compress_tool_outputs(messages, max_tokens=10) == 0


def test_failure_history_keeps_latest_in_full_and_summarizes_older():
    history = [failure(1, "first problem", "OLD OUTPUT"), failure(2, "second problem", "NEW OUTPUT")]
    text = render_failures(history, max_tokens=1000)
    assert "Attempt 1 (summary)" in text and "first problem" in text
    assert "OLD OUTPUT" not in text
    assert "NEW OUTPUT" in text and "second problem" in text


def test_failure_history_drops_oldest_summaries_to_fit():
    history = [failure(i, f"problem {i} " + "p" * 200) for i in range(1, 6)]
    text = render_failures(history, max_tokens=200)
    assert "Attempt 5" in text and "Attempt 1" not in text
    assert estimate_tokens(text) <= 200 + 10
