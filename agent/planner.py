from agent.prompts import planner as prompts
from agent.schemas import Plan
from agent.state import PatchPilotState


def build_planner_message(state: PatchPilotState) -> str:
    files = "\n".join(state.repo_metadata.files)
    return f"## Issue\n{state.issue}\n\n## Repository files\n{files}"


def plan(state: PatchPilotState, llm) -> dict:
    messages = [("system", prompts.SYSTEM), ("human", build_planner_message(state))]
    result: Plan = llm.with_structured_output(Plan, method="json_schema").invoke(messages)

    # Models sometimes invent paths; keep only files that actually exist.
    known = set(state.repo_metadata.files)
    existing = [path for path in result.relevant_files if path in known]
    if not existing:
        raise ValueError(f"planner selected no files that exist in the repo: {result.relevant_files}")
    return {"plan": result.model_copy(update={"relevant_files": existing})}
