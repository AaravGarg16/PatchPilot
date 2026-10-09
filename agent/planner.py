from pathlib import Path

from agent.context import ContextBudget, build_code_context, fit
from agent.prompts import planner as prompts
from agent.schemas import Plan
from agent.state import PatchPilotState
from agent.structured import StructuredOutputError, call_structured
from tools.filesystem import is_test_file


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


def fallback_plan(state: PatchPilotState) -> Plan:
    """A minimal valid plan when the planner's reply cannot be parsed: the issue itself and the files retrieval found."""
    known = set(state.repo_metadata.files)
    return Plan(
        problem=fit(state.issue.strip(), 150),
        relevant_files=[f for f in state.retrieved_files if f in known and not is_test_file(f)],
        implementation_steps=[],
        tests_needed=[],
        risks=["The planner's output could not be parsed, so this is a fallback plan built from retrieval alone."],
    )


def plan(state: PatchPilotState, llm) -> dict:
    messages = [("system", prompts.SYSTEM), ("human", build_planner_message(state))]
    try:
        result, retried = call_structured(llm, Plan, messages, run_name="planner")
    except StructuredOutputError as e:
        # One unparseable reply should not end the task: the executor can still work from the issue and its tools.
        note = f"fallback: planner used a minimal plan ({e})"
        return {"plan": fallback_plan(state), "degradations": [*state.degradations, note]}
    notes = [*state.degradations, "retry: planner, after an unparseable reply"] if retried else state.degradations

    # Models sometimes invent paths; keep only files that actually exist.
    known = set(state.repo_metadata.files)
    existing = [path for path in result.relevant_files if path in known]
    if not existing:
        raise ValueError(f"planner selected no files that exist in the repo: {result.relevant_files}")
    return {"plan": result.model_copy(update={"relevant_files": existing}), "degradations": notes}
