import tempfile
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agent import target
from agent.executor import implement
from agent.planner import plan
from agent.state import PatchPilotState, RepoMetadata
from tools import git, testing


def analyze_repository(state: PatchPilotState, workspace_root: str | None) -> dict:
    workspace = Path(tempfile.mkdtemp(prefix="patchpilot-", dir=workspace_root))
    repo = git.checkout_repo(state.repo_id, workspace / "repo")
    return {"repo_metadata": RepoMetadata(path=str(repo), files=git.list_files(repo))}


def run_tests(state: PatchPilotState) -> dict:
    return {"test_results": testing.run_tests(Path(state.repo_metadata.path))}


def route_after_tests(state: PatchPilotState) -> str:
    if state.test_results.passed or state.attempt_number >= state.max_attempts:
        return "finalize"
    return "implement"


def finalize(state: PatchPilotState) -> dict:
    return {"diff": git.diff(Path(state.repo_metadata.path)), "success": state.test_results.passed}


def build_graph(llm, workspace_root: str | None = None):
    graph = StateGraph(PatchPilotState)
    graph.add_node("analyze_repository", lambda s: analyze_repository(s, workspace_root))
    graph.add_node("plan", lambda s: plan(s, llm))
    graph.add_node("implement", lambda s: implement(s, llm))
    graph.add_node("run_tests", run_tests)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "analyze_repository")
    graph.add_edge("analyze_repository", "plan")
    graph.add_edge("plan", "implement")
    graph.add_edge("implement", "run_tests")
    graph.add_conditional_edges("run_tests", route_after_tests, {"implement": "implement", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()


def main():
    from agent.llm import get_chat_model, load_llm_settings

    settings = load_llm_settings()
    print(f"Model: {settings.provider}/{settings.model}")
    graph = build_graph(get_chat_model(settings))

    final: dict = {}
    for update in graph.stream(PatchPilotState(repo_id=target.REPO, issue=target.ISSUE), stream_mode="updates"):
        for node, values in update.items():
            final.update(values or {})
            if node == "analyze_repository":
                print(f"[{node}] workspace {final['repo_metadata'].path}")
            elif node == "plan":
                print(f"[{node}] {final['plan'].problem}\n        files: {', '.join(final['plan'].relevant_files)}")
            elif node == "implement":
                print(f"[{node}] attempt {final['attempt_number']}: {final['patch_summary']}")
                for reason in final["rejected_edits"]:
                    print(f"        rejected: {reason}")
            elif node == "run_tests":
                result = final["test_results"]
                print(f"[{node}] {'passed' if result.passed else 'failed'}: {result.output.strip().splitlines()[-1]}")

    print(f"\nResult: {'SUCCESS' if final['success'] else 'FAILED'} after {final['attempt_number']} attempt(s)\n")
    print(final["diff"] or "(no changes)")


if __name__ == "__main__":
    main()
