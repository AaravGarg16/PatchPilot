from pathlib import Path

from agent.context import ContextBudget, build_code_context, fit
from agent.prompts import planner as prompts
from agent.schemas import Plan
from agent.state import PatchPilotState


def build_planner_message(state: PatchPilotState) -> str:
    budget = ContextBudget(total_tokens=state.context_budget)
    code = build_code_context(
        Path(state.repo_metadata.path), state.retrieved_files, state.retrieved_chunks, budget.tokens("code")
    )
    # The planner writes the plan rather than reading one, so the file list gets the plan share.
    files = fit("\n".join(state.repo_metadata.files), budget.tokens("plan"))
    return (
        f"## Issue\n{fit(state.issue, budget.tokens('task'))}\n\n"
        f"## Retrieved code\n{code}\n\n"
        f"## Repository files\n{files}"
    )


def plan(state: PatchPilotState, llm) -> dict:
    messages = [("system", prompts.SYSTEM), ("human", build_planner_message(state))]
    result: Plan = llm.with_structured_output(Plan, method="json_schema").invoke(messages)

    # Models sometimes invent paths; keep only files that actually exist.
    known = set(state.repo_metadata.files)
    existing = [path for path in result.relevant_files if path in known]
    if not existing:
        raise ValueError(f"planner selected no files that exist in the repo: {result.relevant_files}")
    return {"plan": result.model_copy(update={"relevant_files": existing})}
