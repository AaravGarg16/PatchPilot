import asyncio
import tempfile
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agent import target
from agent.evaluator import evaluate
from agent.executor import implement
from agent.planner import plan
from agent.state import PatchPilotState, RepoMetadata
from tools import git, linting, testing


def analyze_repository(state: PatchPilotState, workspace_root: str | None) -> dict:
    workspace = Path(tempfile.mkdtemp(prefix="patchpilot-", dir=workspace_root))
    repo = git.checkout_repo(state.repo_id, workspace / "repo")
    return {"repo_metadata": RepoMetadata(path=str(repo), files=git.list_files(repo))}


def run_checks(state: PatchPilotState) -> dict:
    # The executor can run these itself through MCP; this run is the authoritative one.
    repo = Path(state.repo_metadata.path)
    return {
        "test_results": testing.run_tests(repo),
        "lint_results": linting.run_linter(repo, state.modified_files),
    }


def route_after_evaluation(state: PatchPilotState) -> str:
    if state.success or state.attempt_number >= state.max_attempts:
        return "finalize"
    return "implement"


def finalize(state: PatchPilotState) -> dict:
    return {"diff": git.diff(Path(state.repo_metadata.path))}


def build_graph(llm, workspace_root: str | None = None):
    async def implement_node(state: PatchPilotState) -> dict:
        return await implement(state, llm)

    graph = StateGraph(PatchPilotState)
    graph.add_node("analyze_repository", lambda s: analyze_repository(s, workspace_root))
    graph.add_node("plan", lambda s: plan(s, llm))
    graph.add_node("implement", implement_node)
    graph.add_node("run_checks", run_checks)
    graph.add_node("evaluate", lambda s: evaluate(s, llm))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "analyze_repository")
    graph.add_edge("analyze_repository", "plan")
    graph.add_edge("plan", "implement")
    graph.add_edge("implement", "run_checks")
    graph.add_edge("run_checks", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluation, {"implement": "implement", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()


async def run_target():
    from agent.llm import get_chat_model, load_llm_settings

    settings = load_llm_settings()
    print(f"Model: {settings.provider}/{settings.model}")
    graph = build_graph(get_chat_model(settings))

    final: dict = {}
    initial = PatchPilotState(repo_id=target.REPO, issue=target.ISSUE)
    async for update in graph.astream(initial, stream_mode="updates"):
        for node, values in update.items():
            final.update(values or {})
            if node == "analyze_repository":
                print(f"[{node}] workspace {final['repo_metadata'].path}")
            elif node == "plan":
                print(f"[{node}] {final['plan'].problem}\n        files: {', '.join(final['plan'].relevant_files)}")
            elif node == "implement":
                print(
                    f"[{node}] attempt {final['attempt_number']}: {final['patch_summary']}\n"
                    f"        tool calls so far: {final['tool_calls']} ({final['tool_errors']} errors)"
                )
            elif node == "run_checks":
                tests, lint = final["test_results"], final["lint_results"]
                last_line = tests.output.strip().splitlines()[-1] if tests.output.strip() else ""
                print(f"[{node}] tests {'passed' if tests.passed else 'failed'}: {last_line}; lint {'clean' if lint.passed else 'problems'}")
            elif node == "evaluate":
                ev = final["evaluation"]
                print(f"[{node}] {ev.recommended_action} (correct={ev.correct}, requirements_met={ev.requirements_met})")
                for problem in ev.problems:
                    print(f"        - {problem}")

    print(f"\nResult: {'SUCCESS' if final['success'] else 'FAILED'} after {final['attempt_number']} attempt(s)\n")
    print(final["diff"] or "(no changes)")


def main():
    asyncio.run(run_target())


if __name__ == "__main__":
    main()
