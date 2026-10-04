from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.context import (
    ContextBudget,
    build_code_context,
    compress_tool_outputs,
    estimate_tokens,
    fit,
    render_failures,
)
from agent.prompts import executor as prompts
from agent.state import PatchPilotState
from tools import git
from tools.mcp_client import repo_tools

# Model turns per attempt. Stops a confused executor from looping forever on tool calls.
MAX_STEPS = 25
# The full-history baseline has no budget; a budget this large never trims anything.
UNLIMITED_TOKENS = 10**9


def _budget(state: PatchPilotState) -> ContextBudget:
    dynamic = state.agent_config.context_mode == "dynamic"
    return ContextBudget(total_tokens=state.context_budget if dynamic else UNLIMITED_TOKENS)


def build_executor_message(state: PatchPilotState, current_diff: str) -> str:
    budget = _budget(state)
    repo = Path(state.repo_metadata.path)
    parts = [f"## Issue\n{fit(state.issue, budget.tokens('task'))}"]
    if state.plan:
        parts.append(f"## Plan\n{fit(state.plan.model_dump_json(indent=2), budget.tokens('plan'))}")

    code_tokens = budget.tokens("code")
    if state.failure_history:
        # Diff-aware context: the diff shares the code budget, since the files below already contain it.
        diff = fit(current_diff, code_tokens // 4, keep="head")
        code_tokens -= estimate_tokens(diff)
        parts.append(
            f"## Earlier attempts failed ({state.attempt_number} of {state.max_attempts} used)\n"
            + render_failures(state.failure_history, budget.tokens("failures"))
        )
        parts.append(f"## Current diff (already applied)\n```diff\n{diff}\n```")

    files = [*(state.plan.relevant_files if state.plan else []), *state.retrieved_files]
    if files or state.retrieved_chunks:
        parts.append(f"## Retrieved code\n{build_code_context(repo, files, state.retrieved_chunks, code_tokens)}")
    else:
        parts.append("## Code\nNothing was retrieved for you. Explore the repository with the tools.")
    return "\n\n".join(parts)


def build_retry_message(state: PatchPilotState, current_diff: str) -> str:
    """Full-history mode: the transcript already holds everything earlier; only the new failure is added."""
    latest = render_failures(state.failure_history[-1:], UNLIMITED_TOKENS)
    return (
        f"## Attempt {state.attempt_number} of {state.max_attempts} failed\n{latest}\n\n"
        f"## Current diff (already applied)\n```diff\n{current_diff}\n```\n\nFix the remaining problems."
    )


async def implement(state: PatchPilotState, llm) -> dict:
    repo = Path(state.repo_metadata.path)
    full_history = state.agent_config.context_mode == "full_history"
    tool_budget = _budget(state).tokens("tool_outputs")
    calls = errors = elided = 0
    summary = "executor stopped without a summary"
    attempt = state.attempt_number + 1
    run_config = {"run_name": f"executor attempt {attempt}", "metadata": {"attempt": attempt}}

    async with repo_tools(repo, state.sandbox_id, state.checks.model_dump_json()) as tools:
        by_name = {tool.name: tool for tool in tools}
        model = llm.bind_tools(tools)
        current_diff = git.diff(repo)
        if full_history and state.executor_history:
            messages = [*state.executor_history, HumanMessage(build_retry_message(state, current_diff))]
        else:
            messages = [SystemMessage(prompts.SYSTEM), HumanMessage(build_executor_message(state, current_diff))]

        for _ in range(MAX_STEPS):
            if not full_history:
                elided += compress_tool_outputs(messages, tool_budget)
            reply: AIMessage = await model.ainvoke(messages, config=run_config)
            messages.append(reply)
            if not reply.tool_calls:
                summary = reply.text or summary
                break
            for call in reply.tool_calls:
                calls += 1
                tool = by_name.get(call["name"])
                if tool is None:
                    result = ToolMessage(f"unknown tool: {call['name']}", tool_call_id=call["id"], status="error")
                else:
                    result = await tool.ainvoke(call)
                if result.status == "error":
                    errors += 1
                messages.append(result)
        else:
            summary = f"executor hit the {MAX_STEPS}-step limit before finishing"

    return {
        "attempt_number": attempt,
        "modified_files": git.changed_files(repo),
        "patch_summary": summary,
        "tool_calls": state.tool_calls + calls,
        "tool_errors": state.tool_errors + errors,
        "elided_tool_outputs": state.elided_tool_outputs + elided,
        "executor_history": messages if full_history else [],
    }
