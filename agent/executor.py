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


def build_executor_message(state: PatchPilotState, current_diff: str) -> str:
    budget = ContextBudget(total_tokens=state.context_budget)
    repo = Path(state.repo_metadata.path)
    parts = [
        f"## Issue\n{fit(state.issue, budget.tokens('task'))}",
        f"## Plan\n{fit(state.plan.model_dump_json(indent=2), budget.tokens('plan'))}",
    ]

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

    files = [*state.plan.relevant_files, *state.retrieved_files]
    parts.append(f"## Retrieved code\n{build_code_context(repo, files, state.retrieved_chunks, code_tokens)}")
    return "\n\n".join(parts)


async def implement(state: PatchPilotState, llm) -> dict:
    repo = Path(state.repo_metadata.path)
    tool_budget = ContextBudget(total_tokens=state.context_budget).tokens("tool_outputs")
    calls = errors = elided = 0
    summary = "executor stopped without a summary"

    async with repo_tools(repo) as tools:
        by_name = {tool.name: tool for tool in tools}
        model = llm.bind_tools(tools)
        messages = [
            SystemMessage(prompts.SYSTEM),
            HumanMessage(build_executor_message(state, git.diff(repo))),
        ]

        for _ in range(MAX_STEPS):
            elided += compress_tool_outputs(messages, tool_budget)
            reply: AIMessage = await model.ainvoke(messages)
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
        "attempt_number": state.attempt_number + 1,
        "modified_files": git.changed_files(repo),
        "patch_summary": summary,
        "tool_calls": state.tool_calls + calls,
        "tool_errors": state.tool_errors + errors,
        "elided_tool_outputs": state.elided_tool_outputs + elided,
    }
