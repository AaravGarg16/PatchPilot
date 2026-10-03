import asyncio
import tempfile
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agent import target
from agent.evaluator import evaluate
from agent.executor import implement
from agent.planner import plan
from agent.state import PatchPilotState, RepoMetadata
from retrieval.retriever import Retriever, select_files
from tools import git, linting, testing

RETRIEVE_TOP_K = 20
MAX_SELECTED_FILES = 5


def analyze_repository(state: PatchPilotState, workspace_root: str | None) -> dict:
    workspace = Path(tempfile.mkdtemp(prefix="patchpilot-", dir=workspace_root))
    repo = git.checkout_repo(state.repo_id, workspace / "repo")
    # The workspace name is unique per run, so it doubles as the run's index id.
    return {"repo_metadata": RepoMetadata(path=str(repo), files=git.list_files(repo)), "index_id": workspace.name}


def index_repository(state: PatchPilotState, retriever: Retriever) -> dict:
    count = retriever.index_paths(state.index_id, Path(state.repo_metadata.path), state.repo_metadata.files)
    return {"indexed_chunks": count}


def _retrieve(retriever: Retriever, state: PatchPilotState, query: str) -> dict:
    chunks = retriever.search(state.index_id, query, top_k=RETRIEVE_TOP_K)
    return {"retrieved_chunks": chunks, "retrieved_files": select_files(chunks, MAX_SELECTED_FILES)}


def retrieve_context(state: PatchPilotState, retriever: Retriever) -> dict:
    return _retrieve(retriever, state, state.issue)


def refresh_context(state: PatchPilotState, retriever: Retriever) -> dict:
    """Before a retry: re-index edited files so chunks match the code, and search again using the evaluator's problems."""
    repo = Path(state.repo_metadata.path)
    retriever.index_paths(state.index_id, repo, git.changed_files(repo))
    problems = "\n".join(state.failure_history[-1].problems) if state.failure_history else ""
    return _retrieve(retriever, state, f"{state.issue}\n{problems}".strip())


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
    return "refresh_context"


def finalize(state: PatchPilotState, retriever: Retriever) -> dict:
    retriever.drop(state.index_id)
    return {"diff": git.diff(Path(state.repo_metadata.path))}


def build_graph(llm, retriever: Retriever, workspace_root: str | None = None):
    async def implement_node(state: PatchPilotState) -> dict:
        return await implement(state, llm)

    graph = StateGraph(PatchPilotState)
    graph.add_node("analyze_repository", lambda s: analyze_repository(s, workspace_root))
    graph.add_node("index_repository", lambda s: index_repository(s, retriever))
    graph.add_node("retrieve_context", lambda s: retrieve_context(s, retriever))
    graph.add_node("plan", lambda s: plan(s, llm))
    graph.add_node("implement", implement_node)
    graph.add_node("run_checks", run_checks)
    graph.add_node("evaluate", lambda s: evaluate(s, llm))
    graph.add_node("refresh_context", lambda s: refresh_context(s, retriever))
    graph.add_node("finalize", lambda s: finalize(s, retriever))

    graph.add_edge(START, "analyze_repository")
    graph.add_edge("analyze_repository", "index_repository")
    graph.add_edge("index_repository", "retrieve_context")
    graph.add_edge("retrieve_context", "plan")
    graph.add_edge("plan", "implement")
    graph.add_edge("implement", "run_checks")
    graph.add_edge("run_checks", "evaluate")
    graph.add_conditional_edges(
        "evaluate", route_after_evaluation, {"refresh_context": "refresh_context", "finalize": "finalize"}
    )
    graph.add_edge("refresh_context", "implement")
    graph.add_edge("finalize", END)
    return graph.compile()


def _describe(node: str, final: dict) -> str | None:
    if node == "analyze_repository":
        return f"workspace {final['repo_metadata'].path}"
    if node == "index_repository":
        return f"{final['indexed_chunks']} chunks from {len(final['repo_metadata'].files)} files"
    if node in ("retrieve_context", "refresh_context"):
        return f"{len(final['retrieved_chunks'])} chunks; top files: {', '.join(final['retrieved_files'])}"
    if node == "plan":
        return f"{final['plan'].problem}\n        files: {', '.join(final['plan'].relevant_files)}"
    if node == "implement":
        return (
            f"attempt {final['attempt_number']}: {final['patch_summary']}\n"
            f"        tool calls so far: {final['tool_calls']} ({final['tool_errors']} errors, "
            f"{final['elided_tool_outputs']} outputs elided)"
        )
    if node == "run_checks":
        tests, lint = final["test_results"], final["lint_results"]
        last_line = tests.output.strip().splitlines()[-1] if tests.output.strip() else ""
        return f"tests {'passed' if tests.passed else 'failed'}: {last_line}; lint {'clean' if lint.passed else 'problems'}"
    if node == "evaluate":
        ev = final["evaluation"]
        lines = [f"{ev.recommended_action} (correct={ev.correct}, requirements_met={ev.requirements_met})"]
        return "\n".join([*lines, *(f"        - {p}" for p in ev.problems)])
    return None


async def run_target():
    from agent.llm import get_chat_model, load_llm_settings
    from retrieval.embeddings import load_embedder
    from retrieval.store import ChunkStore, database_url

    settings = load_llm_settings()
    embedder = load_embedder()
    print(f"Model: {settings.provider}/{settings.model}; embeddings: {embedder.model_name} (local)")
    retriever = Retriever(ChunkStore(database_url(), embedder.dim), embedder)
    graph = build_graph(get_chat_model(settings), retriever)

    final: dict = {}
    initial = PatchPilotState(repo_id=target.REPO, issue=target.ISSUE)
    async for update in graph.astream(initial, stream_mode="updates"):
        for node, values in update.items():
            final.update(values or {})
            line = _describe(node, final)
            if line:
                print(f"[{node}] {line}")

    print(f"\nResult: {'SUCCESS' if final['success'] else 'FAILED'} after {final['attempt_number']} attempt(s)\n")
    print(final["diff"] or "(no changes)")


def main():
    asyncio.run(run_target())


if __name__ == "__main__":
    main()
