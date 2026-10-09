"""Malformed model output and unexpected exceptions degrade a run; they do not crash it or the batch."""
import asyncio
import subprocess

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage

import agent.run as run_module
import evals.run as eval_run
from agent.errors import describe_error, is_fatal
from agent.executor import REPLY_REMINDER
from agent.planner import fallback_plan
from agent.schemas import Evaluation, Plan
from agent.state import PatchPilotState, RepoMetadata
from agent.structured import (
    PARSE_ERRORS,
    RETRY_REMINDER,
    STRUCTURED_MAX_TOKENS,
)
from evals import results as store
from evals.metrics import summarize
from evals.results import TaskResult
from evals.tasks import Task
from tests.fakes import FIX_PAGINATE, FIX_TOTAL_PAGES, FakeLLM, done
from tests.test_graph import FINISH, PLAN

CUT_OFF = OutputParserException("Failed to parse Plan from completion {...}. Got: 2 validation errors: tests_needed, risks")
FIX = [FIX_PAGINATE, FIX_TOTAL_PAGES, done()]


# --- 1. planner ------------------------------------------------------------------------------------------------------


def test_a_cut_off_plan_raises_the_error_we_catch(monkeypatch):
    """The reported failure, through the real parser: JSON cut off before tests_needed and risks."""
    from langchain_anthropic import ChatAnthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used")
    parser = ChatAnthropic(model="claude-sonnet-5").with_structured_output(Plan, method="json_schema").last
    truncated = '{"problem": "x", "relevant_files": ["a.py"], "implementation_steps": ["fix it"], "tests_nee'
    with pytest.raises(PARSE_ERRORS, match="tests_needed"):
        parser.invoke(AIMessage(content=truncated))


def test_planner_retries_once_with_a_reminder(run_report):
    llm = FakeLLM({Plan: [CUT_OFF, PLAN], Evaluation: [FINISH]}, executor_turns=FIX)

    report = run_report(llm)

    assert report.success and report.error is None
    assert report.degradations == ["retry: planner, after an unparseable reply"]
    first, second = [messages for schema, messages in llm.structured_calls if schema is Plan]
    assert second[: len(first)] == first and second[-1] == ("human", RETRY_REMINDER)
    # More output room than the model default, which forces streaming on the Anthropic SDK.
    assert llm.structured_kwargs[0] == {"max_tokens": STRUCTURED_MAX_TOKENS, "stream": True}


def test_planner_falls_back_to_a_minimal_plan_instead_of_failing_the_task(run_report):
    llm = FakeLLM({Plan: [CUT_OFF], Evaluation: [FINISH]}, executor_turns=FIX)  # the planner never parses

    report = run_report(llm)

    assert report.error is None and report.success  # the task still ran, and here still got fixed
    plan = report.state.plan
    assert plan.relevant_files and "shop/pagination.py" in plan.relevant_files
    assert not any(f.startswith("tests/") for f in plan.relevant_files)
    assert "fallback plan" in plan.risks[0]
    assert len([s for s, _ in llm.structured_calls if s is Plan]) == 2  # asked twice, not forever
    assert report.degradations[0].startswith("fallback: planner used a minimal plan")


def test_fallback_plan_is_valid_even_with_nothing_retrieved(tmp_path):
    state = PatchPilotState(issue="It breaks.\n" + "x" * 5000, repo_metadata=RepoMetadata(path=str(tmp_path), files=[]))
    plan = fallback_plan(state)
    assert plan.relevant_files == [] and plan.problem.startswith("It breaks.") and len(plan.problem) < 1000


def test_real_model_honours_the_raised_limit_and_streams(monkeypatch):
    """Checked on ChatAnthropic's own request builder and streaming rule; nothing is invoked, so nothing is sent."""
    from langchain_anthropic import ChatAnthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used")
    llm = ChatAnthropic(model="claude-sonnet-5", max_tokens=16_000)
    payload = llm._get_request_payload([HumanMessage("plan it")], max_tokens=STRUCTURED_MAX_TOKENS)
    assert payload["max_tokens"] == STRUCTURED_MAX_TOKENS  # the per-call value wins over the model default
    assert llm._should_stream(async_api=False, stream=True) is True
    # The SDK would refuse this limit without streaming, which is why call_structured asks for it.
    with pytest.raises(ValueError, match="Streaming is required"):
        llm._client._calculate_nonstreaming_timeout(STRUCTURED_MAX_TOKENS, None)


def test_batch_mode_ignores_the_streaming_request(monkeypatch):
    from agent.batching import BatchChatAnthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used")
    llm = BatchChatAnthropic(model="claude-sonnet-5")
    assert llm._should_stream(async_api=False, stream=True) is False
    assert "stream" not in llm._batch_params([HumanMessage("hi")], None, stream=True, max_tokens=STRUCTURED_MAX_TOKENS)


# --- evaluator (same failure class) ----------------------------------------------------------------------------------


def test_evaluator_without_a_verdict_defers_to_passing_tests(run_report):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [CUT_OFF]}, executor_turns=FIX)

    report = run_report(llm)

    assert report.error is None and report.success
    assert report.degradations == [report.degradations[0]] and report.degradations[0].startswith(
        "fallback: evaluator attempt 1 gave no verdict, deferred to the tests")


def test_evaluator_without_a_verdict_never_accepts_failing_tests(run_report):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [CUT_OFF]}, executor_turns=[FIX_PAGINATE, done()])

    report = run_report(llm, max_attempts=1)

    assert report.error is None and not report.success and not report.state.test_results.passed


# --- 2. executor -----------------------------------------------------------------------------------------------------


def test_executor_retries_an_unparseable_reply_and_carries_on(run_report):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, CUT_OFF, FIX_TOTAL_PAGES, done()])

    report = run_report(llm)

    assert report.error is None and report.success and report.attempts == 1
    assert report.degradations == ["retry: executor attempt 1, after an unparseable reply"]
    retry_call = llm.executor_calls[2]
    assert retry_call[-1].content == REPLY_REMINDER


def test_two_unparseable_replies_end_the_attempt_not_the_task(run_report):
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[CUT_OFF, CUT_OFF, *FIX])

    report = run_report(llm)

    assert report.error is None  # previously this was an agent error for the whole task
    assert report.attempts == 2 and report.success  # attempt 1 ended early, was checked, and attempt 2 fixed it
    assert report.degradations[0] == "fallback: executor attempt 1 ended early after two unparseable replies"
    (failed_attempt,) = report.state.failure_history  # the early-ended attempt was checked like any other
    assert failed_attempt.attempt == 1 and not failed_attempt.tests_passed


def test_a_tool_call_with_invalid_json_arguments_is_retried(run_report):
    broken = AIMessage(content="", invalid_tool_calls=[
        {"name": "edit_file", "args": '{"path": "shop/pagination.py", "old_str', "id": "call_x", "error": "cut off"}])
    llm = FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[broken, *FIX])

    report = run_report(llm)

    assert report.error is None and report.success
    assert report.degradations == ["retry: executor attempt 1, after an unparseable reply"]


# --- 3. unexpected crashes -------------------------------------------------------------------------------------------


def test_errors_are_described_by_their_cause_not_the_wrapper():
    group = ExceptionGroup("unhandled errors in a TaskGroup", [
        ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("connection reset")]),
    ])
    assert describe_error(group) == "ConnectionError: connection reset (raised inside a task group)"
    assert describe_error(ValueError("bad")) == "ValueError: bad"
    assert not is_fatal(group)
    assert is_fatal(BaseExceptionGroup("stop", [KeyboardInterrupt()]))  # never swallowed


class FailingExecutor(FakeLLM):
    """The model call fails while the executor's MCP tool session is open, as on django-16910."""

    def bind_tools(self, tools):
        class Caller:
            async def ainvoke(self, messages, config=None, **kwargs):
                raise ConnectionError("simulated API failure mid-attempt")

        return Caller()


def containers_for(run_id: str) -> str:
    return subprocess.run(["docker", "ps", "-aq", "--filter", f"label=patchpilot.run={run_id}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_a_crash_inside_the_tool_session_is_recorded_with_its_real_reason(retriever, sandbox_image, tmp_path):
    from tests.fakes import FAKE_MODEL, FIXTURE_REPO

    llm = FailingExecutor({Plan: [PLAN], Evaluation: [FINISH]})
    report = asyncio.run(run_module.run_issue(
        FIXTURE_REPO, "Pagination skips the first page", llm=llm, model_name=FAKE_MODEL, retriever=retriever,
        workspace_root=str(tmp_path), raise_errors=False))

    assert report.error == "ConnectionError: simulated API failure mid-attempt (raised inside a task group)"
    assert not report.success and report.llm_calls == 1 and report.input_tokens > 0  # the planner's cost is kept
    assert containers_for(report.run_id) == ""


def test_a_cleanup_failure_does_not_spoil_a_finished_run(run_report, monkeypatch, caplog):
    def broken_cleanup(run_id):
        raise RuntimeError("docker daemon went away")

    monkeypatch.setattr(run_module, "destroy_run", broken_cleanup)
    report = run_report(FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=FIX))

    assert report.success and report.error is None
    assert "cleanup after run" in caplog.text and "docker daemon went away" in caplog.text


def test_one_crashing_task_does_not_take_down_the_batch(monkeypatch, tmp_path):
    async def fake_run_task(task, config, **kwargs):
        if task.instance_id == "t__t-1":
            raise ExceptionGroup("unhandled errors in a TaskGroup", [ValueError("boom in task 1")])
        return TaskResult(instance_id=task.instance_id, source="custom", config=config, status="done", cost_usd=0.1)

    monkeypatch.setattr(eval_run, "run_task", fake_run_task)
    tasks = [Task(instance_id=f"t__t-{i}", source="custom", repo="t/t", problem_statement="x", repo_path=".")
             for i in range(3)]
    done_runs = asyncio.run(eval_run.run_suite(tasks, ["full"], tmp_path, llm=None, model_name="m", retriever=None,
                                               parallel=2, log=lambda _: None))

    saved = {r.instance_id: r for r in store.load_all(tmp_path / "full")}
    assert len(done_runs) == 3 and len(saved) == 3
    assert saved["t__t-1"].status == "agent_error"
    assert saved["t__t-1"].error == "unexpected crash outside the agent run: ValueError: boom in task 1 (raised inside a task group)"
    assert saved["t__t-0"].status == saved["t__t-2"].status == "done"


def test_fallback_runs_are_counted_separately_from_retries():
    def result(i, notes):
        return TaskResult(instance_id=f"t{i}", source="swebench", config="full", status="done", degradations=notes)

    summary = summarize("full", [result(0, []), result(1, ["retry: planner, after an unparseable reply"]),
                                 result(2, ["fallback: planner used a minimal plan (...)"])])
    assert summary.fallback_runs == 1
