import asyncio
import subprocess

import pytest

from agent import executor
from agent.run import run_issue
from agent.schemas import Evaluation, Plan
from tests.fakes import (
    FAKE_MODEL,
    FAKE_USAGE,
    FIX_PAGINATE,
    FIX_TOTAL_PAGES,
    FIXTURE_REPO,
    FakeLLM,
    done,
    tool_call,
)

PLAN = Plan(
    problem="paginate is off by one page and total_pages floors instead of rounding up.",
    relevant_files=["shop/pagination.py", "tests/test_pagination.py", "shop/made_up.py"],
    implementation_steps=["Start pages at (page - 1) * per_page", "Round total_pages up"],
    tests_needed=["tests/test_pagination.py"],
    risks=["Callers relying on 0-indexed pages"],
)

FINISH = Evaluation(correct=True, requirements_met=True, tests_pass=True, problems=[], recommended_action="finish")


def retry(*problems: str, tests_pass: bool = False) -> Evaluation:
    return Evaluation(
        correct=False, requirements_met=False, tests_pass=tests_pass, problems=list(problems), recommended_action="retry"
    )


@pytest.fixture
def run_report(retriever, sandbox_image, tmp_path):
    def _run(llm, on_update=None, **state):
        return asyncio.run(run_issue(
            FIXTURE_REPO, "Pagination skips the first page", llm=llm, model_name=FAKE_MODEL,
            retriever=retriever, workspace_root=str(tmp_path), on_update=on_update, **state,
        ))

    return _run


@pytest.fixture
def run(run_report):
    """Most tests only care about the final state."""
    return lambda llm, **state: run_report(llm, **state).state


def test_fixes_bug_after_evaluator_feedback(run):
    llm = FakeLLM(
        {Plan: [PLAN], Evaluation: [retry("total_pages still floors 25/10 to 2"), FINISH]},
        executor_turns=[
            tool_call("read_file", path="shop/pagination.py"), FIX_PAGINATE, done("Fixed the page offset."),
            FIX_TOTAL_PAGES, tool_call("run_tests"), done("Rounded total_pages up."),
        ],
    )

    result = run(llm)

    assert result.success is True
    assert result.attempt_number == 2
    assert result.test_results.passed and result.lint_results.passed
    assert "+    start = (page - 1) * per_page" in result.diff
    assert "+    return -(-total_items // per_page)" in result.diff
    assert result.modified_files == ["shop/pagination.py"]
    assert (result.tool_calls, result.tool_errors) == (4, 0)

    # The failure from attempt 1 (test output plus evaluator problems) is fed into attempt 2.
    (history,) = result.failure_history
    assert "test_total_pages_rounds_up" in history.test_output
    assert history.problems == ["total_pages still floors 25/10 to 2"]
    first, second = llm.executor_task_prompts()
    assert "Earlier attempts failed" not in first
    assert "total_pages still floors" in second
    assert "+    start = (page - 1) * per_page" in second  # current diff, so it builds on its own edit
    assert "Earlier failed attempts" in llm.prompts_for(Evaluation)[1]


def test_executor_discovers_tools_over_mcp(run):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, FIX_TOTAL_PAGES, done()])

    run(llm)

    assert llm.bound_tool_names == [
        "create_file", "edit_file", "git_diff", "list_directory", "read_file", "run_linter", "run_tests", "search_code",
    ]


def test_evaluator_can_reject_a_patch_whose_tests_pass(run):
    llm = FakeLLM(
        {Plan: [PLAN], Evaluation: [retry("Remove the unrelated helper", tests_pass=True), FINISH]},
        executor_turns=[
            FIX_PAGINATE, FIX_TOTAL_PAGES, tool_call("create_file", path="shop/helper.py", content="X = 1\n"), done(),
            tool_call("git_diff"), done("Nothing else to change."),
        ],
    )

    result = run(llm)

    assert result.attempt_number == 2
    assert result.success is True
    assert "Remove the unrelated helper" in llm.executor_task_prompts()[1]


def test_failing_tests_overrule_an_evaluator_that_says_finish(run):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, done(), done()])

    result = run(llm, max_attempts=2)

    assert result.success is False
    assert result.attempt_number == 2
    assert not result.test_results.passed
    assert len(llm.prompts_for(Evaluation)) == 2
    assert len(result.failure_history) == 2


def test_tool_errors_go_back_to_the_model_and_are_counted(run):
    cheat = tool_call("edit_file", path="tests/test_pagination.py", old_string="import pytest", new_string="")
    llm = FakeLLM(
        {Plan: [PLAN], Evaluation: [FINISH]},
        executor_turns=[cheat, tool_call("delete_everything"), FIX_PAGINATE, FIX_TOTAL_PAGES, done()],
    )

    result = run(llm)

    assert result.success is True
    assert (result.tool_calls, result.tool_errors) == (4, 2)
    assert "tests/test_pagination.py" not in result.diff
    # What the model saw after each bad call:
    assert "existing test files cannot be modified" in str(llm.executor_calls[1][-1].content)
    assert "unknown tool: delete_everything" in llm.executor_calls[2][-1].content


def test_executor_step_limit_ends_the_attempt(run, monkeypatch):
    monkeypatch.setattr(executor, "MAX_STEPS", 3)
    llm = FakeLLM(
        {Plan: [PLAN], Evaluation: [retry("No changes were made")]},
        executor_turns=[tool_call("list_directory") for _ in range(3)],
    )

    result = run(llm, max_attempts=1)

    assert result.success is False
    assert "step limit" in result.patch_summary
    assert len(llm.executor_calls) == 3


def test_lint_problems_reach_the_evaluator(run):
    unused_import = tool_call(
        "edit_file", path="shop/pagination.py", old_string="def total_pages", new_string="import os\n\n\ndef total_pages"
    )
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, FIX_TOTAL_PAGES, unused_import, done()])

    result = run(llm)

    assert not result.lint_results.passed
    assert "F401" in llm.prompts_for(Evaluation)[0]


def test_planner_gets_retrieved_code_and_file_list(run):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, FIX_TOTAL_PAGES, done()])

    result = run(llm)

    assert result.indexed_chunks > 0
    assert result.retrieved_files[0] == "shop/pagination.py"
    (planner_prompt,) = llm.prompts_for(Plan)
    assert "### shop/pagination.py (whole file)" in planner_prompt
    assert "## Repository files\n" in planner_prompt and "README.md" in planner_prompt
    assert result.plan.relevant_files == ["shop/pagination.py", "tests/test_pagination.py"]


def test_index_is_dropped_when_the_run_ends(run, retriever):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, FIX_TOTAL_PAGES, done()])

    result = run(llm)

    assert retriever.store.count(result.index_id) == 0


def test_retry_refreshes_the_index_with_edited_code(run):
    llm = FakeLLM(
        {Plan: [PLAN], Evaluation: [retry("total_pages still floors"), FINISH]},
        executor_turns=[FIX_PAGINATE, done(), FIX_TOTAL_PAGES, done()],
    )

    result = run(llm)

    (paginate,) = [c for c in result.retrieved_chunks if c.symbol == "paginate"]
    assert "(page - 1) * per_page" in paginate.content
    # The refresh searched with the evaluator's problem, and the retry prompt shows the edited file.
    assert "start = (page - 1) * per_page" in llm.executor_task_prompts()[1].split("## Retrieved code")[1]


def test_old_tool_outputs_are_elided_under_a_tight_budget(run):
    reads = [tool_call("read_file", path=p) for p in ("tests/test_pagination.py", "shop/formatting.py", "README.md")]
    llm = FakeLLM(
        {Plan: [PLAN], Evaluation: [FINISH]},
        executor_turns=[*reads, FIX_PAGINATE, FIX_TOTAL_PAGES, done()],
    )

    result = run(llm, context_budget=3000)

    assert result.success is True
    assert result.elided_tool_outputs >= 1
    last_call = llm.executor_calls[-1]
    assert any(isinstance(m.content, str) and m.content.startswith("[elided read_file") for m in last_call)
    assert len(llm.executor_task_prompts()[0]) <= 3000 * 4


def test_planner_with_no_real_files_fails_loudly(run):
    bad_plan = PLAN.model_copy(update={"relevant_files": ["nope.py"]})
    llm = FakeLLM({Plan: [bad_plan], Evaluation: [FINISH]})

    with pytest.raises(ValueError, match="no files that exist"):
        run(llm)


def containers_for(run_id: str) -> str:
    return subprocess.run(["docker", "ps", "-aq", "--filter", f"label=patchpilot.run={run_id}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_run_report_has_metrics_and_before_after_tests(run_report):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, FIX_TOTAL_PAGES, done()])

    report = run_report(llm)

    assert report.success and report.attempts == 1 and report.retries == 0
    assert (report.tests_before_passed, report.tests_after_passed) == (False, True)
    # planner + 3 executor turns + evaluator, each reporting FAKE_USAGE
    assert report.llm_calls == 5
    assert report.input_tokens == 5 * FAKE_USAGE["input_tokens"]
    assert report.output_tokens == 5 * FAKE_USAGE["output_tokens"]
    assert report.cost_usd == pytest.approx((5000 * 2.00 + 500 * 10.00) / 1_000_000)
    assert report.latency_seconds > 0
    assert report.trace_id is None  # Langfuse is off without keys


def test_sandbox_is_gone_after_a_successful_run(run_report):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, FIX_TOTAL_PAGES, done()])

    report = run_report(llm)

    assert report.state.sandbox_id == f"patchpilot-{report.run_id}"
    assert containers_for(report.run_id) == ""


def test_sandbox_and_index_are_cleaned_up_when_a_run_crashes(run_report, retriever):
    seen = {}
    bad_plan = PLAN.model_copy(update={"relevant_files": ["nope.py"]})
    llm = FakeLLM({Plan: [bad_plan], Evaluation: [FINISH]})

    with pytest.raises(ValueError):
        run_report(llm, on_update=lambda node, final: seen.update(final))

    assert seen["sandbox_id"]  # the container did start before the crash
    assert containers_for(seen["run_id"]) == ""
    assert retriever.store.count(seen["run_id"]) == 0
